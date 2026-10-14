"""
Обёртка над yt-dlp: проверка видео, получение субтитров, скачивание аудио.

Главные функции:
- check_video(url)      — проверяет ссылку и видео (длина, не стрим, не плейлист).
                          Возвращает метаданные, чтобы не запрашивать их повторно.
- get_subtitles(url)    — пытается достать субтитры (бесплатно и быстро).
- download_audio(url)   — скачивает аудио в mp3 для Whisper (если субтитров нет).

Все ошибки, которые стоит показать пользователю, выбрасываются как VideoError
(это подкласс ValueError) с понятным текстом на русском.
"""

import html
import os
import re
import tempfile
import uuid
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import DownloadError

# Языки субтитров, которые мы принимаем (в порядке приоритета).
SUBTITLE_LANGS = ("ru", "en")

# Лимит Whisper API — 25 МБ на файл. Берём с небольшим запасом.
WHISPER_MAX_BYTES = 24 * 1024 * 1024

# Если после очистки субтитров осталось меньше символов — считаем,
# что субтитры бесполезны (например, только "[Музыка]"), и идём в Whisper.
MIN_SUBTITLE_CHARS = 50

# Разрешённые домены. Проверяем сами, потому что yt-dlp умеет качать
# с тысяч сайтов (и даже с внутренних адресов сервера) — нам это не нужно и опасно.
ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class VideoError(ValueError):
    """Ошибка, связанная с видео. Текст можно показывать пользователю как есть."""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _max_duration() -> int:
    """Читаем лимит длины видео из .env при каждом вызове (по умолчанию 1 час)."""
    return int(os.getenv("MAX_VIDEO_DURATION_SEC", "3600"))


def _cookies_file() -> str | None:
    """
    Если в переменной YTDLP_COOKIES лежит содержимое cookies.txt —
    сохраняем его во временный файл и возвращаем путь.
    Это нужно на случай, если YouTube блокирует IP сервера.
    Файл создаётся один раз за время жизни процесса.
    """
    content = os.getenv("YTDLP_COOKIES", "").strip()
    if not content:
        return None
    path = os.path.join(tempfile.gettempdir(), "ytdlp_cookies.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
    return path


def _base_opts() -> dict:
    """Общие настройки yt-dlp для всех вызовов."""
    opts = {
        "quiet": True,  # не засорять логи прогресс-баром
        "no_warnings": True,
        "noplaylist": True,  # ссылка вида watch?v=...&list=... → берём только само видео
        "allowed_extractors": ["youtube"],  # только YouTube, никаких других сайтов
        "socket_timeout": 30,
    }
    cookies = _cookies_file()
    if cookies:
        opts["cookiefile"] = cookies
    return opts


def _validate_url(url: str) -> None:
    """Проверяем, что это действительно ссылка на YouTube."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        raise VideoError("Некорректная ссылка.")
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ALLOWED_HOSTS:
        raise VideoError("Поддерживаются только ссылки на YouTube (youtube.com или youtu.be).")


def _friendly_error(err: Exception) -> VideoError:
    """
    Превращаем техническую ошибку yt-dlp в понятное сообщение.
    Оригинальный текст печатаем в лог — он пригодится для отладки.
    """
    msg = str(err)
    print(f"[downloader] yt-dlp error: {msg}")
    low = msg.lower()

    if "not a bot" in low or ("sign in to confirm you" in low and "age" not in low):
        return VideoError(
            "YouTube требует авторизацию: сервер временно заблокирован YouTube. "
            "Попробуйте позже."
        )
    if "confirm your age" in low or "age-restricted" in low or "inappropriate for some users" in low:
        return VideoError("Видео с возрастным ограничением — его нельзя обработать.")
    if "private video" in low:
        return VideoError("Это приватное видео — доступ к нему закрыт.")
    if "members-only" in low or "join this channel" in low:
        return VideoError("Видео доступно только спонсорам канала.")
    if "premiere" in low or "live event will begin" in low:
        return VideoError("Премьера ещё не началась — попробуйте после выхода видео.")
    if "http error 429" in low or "too many requests" in low:
        return VideoError("YouTube временно ограничил запросы. Попробуйте через несколько минут.")
    if "unsupported url" in low or "is not a valid url" in low or "incomplete youtube id" in low:
        return VideoError("Некорректная ссылка на YouTube-видео.")
    if "not available in your country" in low or ("geo" in low and "restrict" in low):
        return VideoError("Видео недоступно в регионе сервера.")
    if "video unavailable" in low or "has been removed" in low or "does not exist" in low:
        return VideoError("Видео недоступно (удалено или заблокировано).")
    return VideoError("Не удалось получить видео с YouTube. Проверьте ссылку и попробуйте ещё раз.")


# ---------------------------------------------------------------------------
# Проверка видео
# ---------------------------------------------------------------------------


def check_video(url: str) -> dict:
    """
    Получает метаданные видео БЕЗ скачивания и проверяет ограничения:
    - это YouTube;
    - это одно видео, а не плейлист/канал;
    - это не прямой эфир и не будущая премьера;
    - длина не больше MAX_VIDEO_DURATION_SEC.

    Возвращает словарь info от yt-dlp — его можно передать в get_subtitles()
    и download_audio(), чтобы не делать лишний запрос к YouTube.
    """
    _validate_url(url)

    opts = _base_opts()
    opts["skip_download"] = True
    # Формат указываем только аудио: так yt-dlp не ругается на отсутствие видеоформатов.
    opts["format"] = "bestaudio/best"

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise _friendly_error(e)

    if not info:
        raise VideoError("Не удалось получить информацию о видео.")

    # Плейлист или канал — у них _type == 'playlist'
    if info.get("_type") == "playlist":
        raise VideoError("Это ссылка на плейлист или канал. Вставьте ссылку на одно видео.")

    # Прямой эфир или запланированная трансляция
    live_status = info.get("live_status")
    if info.get("is_live") or live_status in ("is_live", "is_upcoming", "post_live"):
        raise VideoError("Прямые эфиры не поддерживаются. Дождитесь окончания и обработки записи.")

    duration = info.get("duration")
    if duration is None:
        # Длина неизвестна — редкий случай. Не блокируем, но пишем в лог.
        print(f"[downloader] Длина видео неизвестна: {url}")
    elif duration > _max_duration():
        max_min = _max_duration() // 60
        raise VideoError(
            f"Видео слишком длинное ({int(duration) // 60} мин). Максимум — {max_min} мин."
        )

    return info


# ---------------------------------------------------------------------------
# Субтитры
# ---------------------------------------------------------------------------


def _pick_subtitle_track(info: dict) -> tuple[str, str] | None:
    """
    Выбирает лучшую дорожку субтитров. Возвращает (язык, url_файла_vtt) или None.

    ВАЖНО: YouTube для любого видео отдаёт "автопереведённые" субтитры на десятки
    языков. Например, у английского видео будет 'ru' — но это машинный перевод
    машинного распознавания, качество плохое. Поэтому берём только:
      1) ручные субтитры автора (info['subtitles']) на ru/en;
      2) автосубтитры на ОРИГИНАЛЬНОМ языке видео (ключ вида 'en-orig'
         или язык, совпадающий с info['language']).
    """

    def vtt_url(tracks: list[dict]) -> str | None:
        for t in tracks or []:
            if t.get("ext") == "vtt" and t.get("url"):
                return t["url"]
        return None

    # 1) Ручные субтитры — лучшее качество
    manual = info.get("subtitles") or {}
    for lang in SUBTITLE_LANGS:
        for key, tracks in manual.items():
            # Ключи бывают 'ru', 'en', 'en-US', 'en-GB' и т.п.
            if key == lang or key.startswith(lang + "-"):
                url = vtt_url(tracks)
                if url:
                    return lang, url

    # 2) Автосубтитры на оригинальном языке
    auto = info.get("automatic_captions") or {}
    original_lang = (info.get("language") or "").split("-")[0]
    for lang in SUBTITLE_LANGS:
        # Ключ '{lang}-orig' — точно оригинальное распознавание
        url = vtt_url(auto.get(f"{lang}-orig"))
        if url:
            return lang, url
        # Иначе берём '{lang}', только если это язык самого видео (не перевод)
        if original_lang == lang:
            url = vtt_url(auto.get(lang))
            if url:
                return lang, url

    return None


def clean_vtt(vtt_text: str) -> str:
    """
    Превращает VTT-файл в чистый текст.

    Что убираем:
    - заголовок WEBVTT и служебные строки (Kind:, Language:, NOTE, STYLE);
    - строки с таймингами "00:00:01.000 --> 00:00:03.000 align:start ...";
    - номера реплик;
    - теги вида <00:00:01.500>, <c>, </c>, <c.colorE5E5E5>;
    - пометки [Музыка], [Аплодисменты], [Music];
    - ДУБЛИ: автосубтитры YouTube "бегущие" — каждая строка повторяется
      2-3 раза в соседних репликах.
    """
    lines_out: list[str] = []
    recent: list[str] = []  # последние добавленные строки — для удаления дублей

    for raw in vtt_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue
        if "-->" in line:  # строка с таймингом
            continue
        if line.isdigit():  # номер реплики
            continue

        line = re.sub(r"<[^>]+>", "", line)  # все теги <...>
        line = html.unescape(line)  # &amp; → &, &gt; → > и т.п.
        line = re.sub(r"\[[^\]]*\]", "", line)  # [Музыка], [Music]
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        # Пропускаем строку, если она совпадает с одной из 3 последних
        if line in recent:
            continue
        lines_out.append(line)
        recent.append(line)
        if len(recent) > 3:
            recent.pop(0)

    return " ".join(lines_out).strip()


def get_subtitles(url: str, info: dict | None = None) -> str | None:
    """
    Пытается получить субтитры видео (ru или en) и вернуть чистый текст.
    Возвращает None, если подходящих субтитров нет — тогда нужен Whisper.

    info — результат check_video(). Если не передан, будет получен заново.
    """
    if info is None:
        info = check_video(url)

    track = _pick_subtitle_track(info)
    if not track:
        print("[downloader] Подходящих субтитров нет")
        return None

    lang, sub_url = track
    try:
        # Скачиваем файл субтитров через yt-dlp: он использует те же cookies
        # и заголовки, что и для видео, поэтому YouTube реже блокирует запрос.
        with yt_dlp.YoutubeDL(_base_opts()) as ydl:
            raw = ydl.urlopen(sub_url).read().decode("utf-8", errors="replace")
    except Exception as e:
        # Субтитры — лишь бесплатный бонус. Если не скачались — не падаем, идём в Whisper.
        print(f"[downloader] Не удалось скачать субтитры: {e}")
        return None

    text = clean_vtt(raw)
    if len(text) < MIN_SUBTITLE_CHARS:
        print(f"[downloader] Субтитры слишком короткие ({len(text)} симв.) — пропускаем")
        return None

    print(f"[downloader] Субтитры получены: язык={lang}, {len(text)} символов")
    return text


# ---------------------------------------------------------------------------
# Скачивание аудио
# ---------------------------------------------------------------------------


def download_audio(url: str, info: dict | None = None) -> tuple[str, float]:
    """
    Скачивает аудиодорожку и конвертирует в mp3: моно, 32 кбит/с, 16 кГц.
    При таком качестве час речи весит ~14 МБ и влезает в лимит Whisper (25 МБ),
    а распознавание речи не страдает (Whisper внутри всё равно работает на 16 кГц).

    Возвращает (путь_к_mp3, длительность_в_секундах).
    Удалять файл после использования — задача вызывающего кода.
    """
    if info is None:
        info = check_video(url)  # проверка длины, стримов, плейлистов

    file_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    out_template = os.path.join(tmp_dir, f"{file_id}.%(ext)s")
    mp3_path = os.path.join(tmp_dir, f"{file_id}.mp3")

    opts = _base_opts()
    opts.update(
        {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            # После скачивания ffmpeg конвертирует файл в mp3 32 кбит/с
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "32",  # число > 10 — это битрейт в кбит/с
                }
            ],
            # Дополнительные параметры ffmpeg: -ac 1 (моно), -ar 16000 (16 кГц)
            "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "16000"]},
        }
    )

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([info.get("webpage_url") or url])
    except DownloadError as e:
        _cleanup_by_prefix(tmp_dir, file_id)
        raise _friendly_error(e)

    if not os.path.exists(mp3_path):
        _cleanup_by_prefix(tmp_dir, file_id)
        raise VideoError("Не удалось извлечь аудио из видео.")

    size = os.path.getsize(mp3_path)
    if size > WHISPER_MAX_BYTES:
        os.remove(mp3_path)
        raise VideoError("Аудио слишком большое для распознавания. Попробуйте видео покороче.")

    duration = float(info.get("duration") or 0)
    print(f"[downloader] Аудио скачано: {mp3_path} ({size / 1024 / 1024:.1f} МБ, {duration:.0f} сек)")
    return mp3_path, duration


def _cleanup_by_prefix(directory: str, prefix: str) -> None:
    """Удаляет недокачанные/промежуточные файлы, если скачивание упало."""
    try:
        for name in os.listdir(directory):
            if name.startswith(prefix):
                os.remove(os.path.join(directory, name))
    except OSError:
        pass
