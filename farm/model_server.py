"""model_server.py — генерация внутри GCP Cloud Run Job (GPU L4) или как HTTP-сервер.

Режим задачи (по умолчанию):
  env DEVICE_ID, SOURCE_IDX, SCENES_URI, VOICE_ID, SPLIT_SCREEN, OUTPUT_URI
  STAGE: "full" | "wan" | "lipsync"   (для "lipsync" нужен CONCAT_URI)
  python model_server.py [--warm]

Режим сервера:
  MODEL_SERVER_TOKEN=... python model_server.py --serve [--warm]

Параметры генерации (env):
  WAN_STEPS=4  WAN_GUIDANCE=1.0  — значения для дистиллированной LightWan;
                                   для обычной Wan 2.2 ставь WAN_STEPS=30..50, WAN_GUIDANCE=5.0
  WAN_WIDTH=480 WAN_HEIGHT=832   — вертикальное видео, кратно 32
  WAN_FPS=24 WAN_MAX_FRAMES=121  WAN_OFFLOAD=1 (выгрузка частей модели в RAM — нужно для L4)

Лицо персонажа (опорный кадр):
  seed = payload.persona_lock → master-сцена.persona_lock → sha256(device_id)[:8] (как раньше).
  Внешность (appearance) в промпт подставляет forge.py — только он видит metadata.json.
  Опорный кадр кэшируется (master_ref_uri в GCS / models/masters локально): после смены
  persona_lock старый кадр нужно удалить, иначе лицо не поменяется.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("model_server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
WAN_DIR = MODELS_DIR / "wan22_ti2v_5b"
XTTS_DIR = MODELS_DIR / "xtts_v2"
LONGCAT_DIR = MODELS_DIR / "longcat_avatar"
SDXL_DIR = MODELS_DIR / "sdxl_turbo"
VOICES_DIR = MODELS_DIR / "voices"
GAMEPLAY_DIR = MODELS_DIR / "gameplay"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


WAN_STEPS = _env_int("WAN_STEPS", 4)
WAN_GUIDANCE = _env_float("WAN_GUIDANCE", 1.0)
WAN_WIDTH = _env_int("WAN_WIDTH", 480) // 32 * 32
WAN_HEIGHT = _env_int("WAN_HEIGHT", 832) // 32 * 32
WAN_FPS = _env_int("WAN_FPS", 24)
WAN_MAX_FRAMES = _env_int("WAN_MAX_FRAMES", 121)
WAN_OFFLOAD = os.getenv("WAN_OFFLOAD", "1") == "1"

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
STAGES = ("full", "wan", "lipsync")

DEFAULT_NEGATIVE = ("extra fingers, deformed hands, distorted face, bad anatomy, cartoon, "
                    "text, watermark, blurry, child, minor, teen")

_load_lock = threading.Lock()
_gpu_lock = threading.Lock()
_WAN_T2V = None
_WAN_I2V = None
_SDXL = None
_XTTS_MODEL = None
_LONGCAT = None
_LONGCAT_TRIED = False
_GCS = None


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

def _parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"ожидался gs://bucket/key, получено: {uri!r}")
    bucket, key = uri[5:].split("/", 1)
    if not bucket or not key:
        raise ValueError(f"некорректный URI: {uri!r}")
    return bucket, key


def _gcs():
    global _GCS
    if _GCS is None:
        from google.cloud import storage
        _GCS = storage.Client()
    return _GCS


def gcs_download(uri: str, local: Path) -> Path:
    bucket, key = _parse_gs(uri)
    local.parent.mkdir(parents=True, exist_ok=True)
    _gcs().bucket(bucket).blob(key).download_to_filename(str(local), timeout=600)
    log.info("downloaded %s -> %s", uri, local)
    return local


def gcs_upload(local: Path, uri: str) -> str:
    bucket, key = _parse_gs(uri)
    _gcs().bucket(bucket).blob(key).upload_from_filename(str(local), timeout=600)
    log.info("uploaded %s -> %s", local, uri)
    return uri


def gcs_exists(uri: str) -> bool:
    bucket, key = _parse_gs(uri)
    return _gcs().bucket(bucket).blob(key).exists()


# ---------------------------------------------------------------------------
# Загрузка моделей (один раз на процесс)
# ---------------------------------------------------------------------------

def _place(pipe):
    if WAN_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def _load_wan():
    """Возвращает (t2v, i2v). i2v может быть None, тогда сцены идут через t2v."""
    global _WAN_T2V, _WAN_I2V
    with _load_lock:
        if _WAN_T2V is not None:
            return _WAN_T2V, _WAN_I2V
        if not WAN_DIR.exists():
            raise FileNotFoundError(f"Wan не найдена: {WAN_DIR}")
        import torch
        from diffusers import WanPipeline
        log.info("загрузка Wan 2.2 из %s", WAN_DIR)
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "local_files_only": True}
        try:
            from diffusers import AutoencoderKLWan
            kwargs["vae"] = AutoencoderKLWan.from_pretrained(
                str(WAN_DIR), subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
        except Exception as e:  # VAE в float32 даёт меньше артефактов, но не обязателен
            log.warning("VAE float32 не загружен (%s), использую из пайплайна", e)
        t2v = WanPipeline.from_pretrained(str(WAN_DIR), **kwargs)
        i2v = None
        try:
            from diffusers import WanImageToVideoPipeline
            i2v = WanImageToVideoPipeline.from_pipe(t2v)  # общие веса, без второй копии в памяти
        except Exception as e:
            log.warning("I2V пайплайн недоступен (%s) — сцены будут text-to-video", e)
        _place(t2v)
        if i2v is not None and not WAN_OFFLOAD:
            i2v.to("cuda")
        elif i2v is not None:
            i2v.enable_model_cpu_offload()
        _WAN_T2V, _WAN_I2V = t2v, i2v
        log.info("Wan 2.2 загружена (steps=%d guidance=%.1f %dx%d)",
                 WAN_STEPS, WAN_GUIDANCE, WAN_WIDTH, WAN_HEIGHT)
        return _WAN_T2V, _WAN_I2V


def _load_sdxl():
    global _SDXL
    with _load_lock:
        if _SDXL is not None or not SDXL_DIR.exists():
            return _SDXL
        import torch
        from diffusers import AutoPipelineForText2Image
        log.info("загрузка SDXL Turbo из %s", SDXL_DIR)
        _SDXL = AutoPipelineForText2Image.from_pretrained(
            str(SDXL_DIR), torch_dtype=torch.float16, local_files_only=True)
        _SDXL.enable_model_cpu_offload()
        return _SDXL


def _load_xtts():
    global _XTTS_MODEL
    with _load_lock:
        if _XTTS_MODEL is not None:
            return _XTTS_MODEL
        if not XTTS_DIR.exists():
            raise FileNotFoundError(f"XTTS не найдена: {XTTS_DIR}")
        log.info("загрузка XTTS v2 из %s", XTTS_DIR)
        from TTS.api import TTS
        cfg_path = XTTS_DIR / "config.json"
        tts = TTS(model_path=str(XTTS_DIR), config_path=str(cfg_path) if cfg_path.exists() else None,
                  progress_bar=False)
        try:
            import torch
            if torch.cuda.is_available():
                tts = tts.to("cuda")
        except Exception as e:
            log.warning("XTTS на GPU не перенесена: %s", e)
        _XTTS_MODEL = tts
        return _XTTS_MODEL


def _load_longcat():
    global _LONGCAT, _LONGCAT_TRIED
    with _load_lock:
        if _LONGCAT is not None or _LONGCAT_TRIED:
            return _LONGCAT
        _LONGCAT_TRIED = True
        if not LONGCAT_DIR.exists():
            log.warning("LongCat не найден (%s) — липсинк пропущен, будет обычное наложение звука",
                        LONGCAT_DIR)
            return None
        try:
            import importlib
            mod = importlib.import_module("longcat_avatar")
            _LONGCAT = mod.AvatarModel.from_pretrained(str(LONGCAT_DIR), device="cuda", int8=True)
            log.info("LongCat загружен")
        except Exception as e:
            log.error("LongCat не загрузился: %s", e)
            _LONGCAT = None
        return _LONGCAT


def warm() -> None:
    _load_wan()
    _load_xtts()
    _load_sdxl()
    _load_longcat()
    log.info("warm: модели загружены")


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def _ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg ошибка: {p.stderr[-1200:]}")


def _duration(path: Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

def _frames_for(duration_sec: float) -> int:
    n = int(round(duration_sec * WAN_FPS))
    n = ((max(5, n) - 1 + 2) // 4) * 4 + 1  # Wan требует num_frames = 4k + 1 (ближайшее)
    max_ok = ((WAN_MAX_FRAMES - 1) // 4) * 4 + 1
    return min(n, max_ok)


def _to_uint8_frames(frames) -> list:
    import numpy as np
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    return list(arr)


def _frames_to_mp4(frames, out_path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=16)
    try:
        for f in _to_uint8_frames(frames):
            writer.append_data(f)
    finally:
        writer.close()


def gen_master_image(prompt: str, negative: str, out_path: Path, seed: int) -> Path:
    import torch
    from PIL import Image
    g = torch.Generator(device="cpu").manual_seed(seed)
    sdxl = _load_sdxl()
    if sdxl is not None:
        img = sdxl(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
                   width=WAN_WIDTH // 8 * 8, height=WAN_HEIGHT // 8 * 8, generator=g).images[0]
    else:
        t2v, _ = _load_wan()
        out = t2v(prompt=prompt, negative_prompt=negative, height=WAN_HEIGHT, width=WAN_WIDTH,
                  num_frames=1, num_inference_steps=WAN_STEPS, guidance_scale=WAN_GUIDANCE,
                  generator=g, output_type="np")
        img = Image.fromarray(_to_uint8_frames(out.frames[0])[0])
    img = img.convert("RGB").resize((WAN_WIDTH, WAN_HEIGHT))
    img.save(out_path)
    log.info("опорный кадр -> %s", out_path)
    return out_path


def gen_scene_video(prompt: str, negative: str, ref_image: Path | None,
                    duration_sec: float, out_path: Path, seed: int) -> Path:
    import torch
    t2v, i2v = _load_wan()
    g = torch.Generator(device="cpu").manual_seed(seed)
    kwargs: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative,
                              "height": WAN_HEIGHT, "width": WAN_WIDTH,
                              "num_frames": _frames_for(duration_sec),
                              "num_inference_steps": WAN_STEPS, "guidance_scale": WAN_GUIDANCE,
                              "generator": g, "output_type": "np"}
    if i2v is not None and ref_image and ref_image.exists():
        from PIL import Image
        kwargs["image"] = Image.open(ref_image).convert("RGB").resize((WAN_WIDTH, WAN_HEIGHT))
        out = i2v(**kwargs)
    else:
        out = t2v(**kwargs)
    frames = out.frames[0]
    _frames_to_mp4(frames, out_path, fps=WAN_FPS)
    log.info("сцена -> %s (%d кадров)", out_path, len(frames))
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out_path


def _detect_lang(text: str) -> str:
    return "ru" if re.search(r"[А-Яа-яЁё]", text or "") else "en"


def synth_voice(text: str, voice_id: str, out_wav: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", voice_id or ""):
        raise ValueError(f"некорректный voice_id: {voice_id!r}")
    speaker_wav = VOICES_DIR / f"{voice_id}.wav"
    if not speaker_wav.exists():
        speaker_wav = VOICES_DIR / "default_speaker.wav"
    if not speaker_wav.exists():
        raise FileNotFoundError(f"нет образца голоса: {VOICES_DIR}/{voice_id}.wav")
    tts = _load_xtts()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    lang = _detect_lang(text)
    tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=lang, file_path=str(out_wav))
    log.info("озвучка -> %s (голос=%s, язык=%s)", out_wav, speaker_wav.name, lang)
    return out_wav


def mux_audio(video: Path, audio: Path, out_video: Path) -> Path:
    # apad + -shortest: длина ролика = длина видео (раньше -shortest обрезал видео до длины фразы)
    _ffmpeg(["-i", str(video), "-i", str(audio),
             "-filter_complex", "[1:a]apad[a]", "-map", "0:v:0", "-map", "[a]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
             "-movflags", "+faststart", str(out_video)])
    log.info("звук наложен -> %s", out_video)
    return out_video


def lipsync(video: Path, audio: Path, out_video: Path) -> Path:
    model = _load_longcat()
    if model is not None:
        try:
            model.synthesize(video_path=str(video), audio_path=str(audio), output_path=str(out_video))
            if out_video.exists() and out_video.stat().st_size > 0:
                log.info("липсинк LongCat -> %s", out_video)
                return out_video
        except Exception as e:
            log.warning("LongCat ошибка (%s), просто накладываю звук", e)
    return mux_audio(video, audio, out_video)


def concat_clips(clips: list[Path], out_video: Path) -> Path:
    if not clips:
        raise ValueError("нет клипов для склейки")
    if len(clips) == 1:
        shutil.copyfile(clips[0], out_video)
        return out_video
    list_file = out_video.with_suffix(".txt")
    list_file.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips),
                         encoding="utf-8")
    try:
        try:
            _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_video)])
        except RuntimeError:
            _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file),
                     "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", str(out_video)])
    finally:
        list_file.unlink(missing_ok=True)
    log.info("склейка -> %s (%d клипов)", out_video, len(clips))
    return out_video


def split_screen(main_video: Path, gameplay: list[Path], out_video: Path, seed: int) -> Path:
    if not gameplay:
        shutil.copyfile(main_video, out_video)
        return out_video
    rnd = random.Random(seed)
    gp = rnd.choice(gameplay)
    dur = _duration(main_video) or 10.0
    gp_dur = _duration(gp)
    offset = rnd.uniform(0, max(0.0, gp_dur - dur - 1)) if gp_dur > dur + 1 else 0.0
    fc = ("[0:v]scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960,setsar=1[top];"
          "[1:v]scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960,setsar=1[bottom];"
          "[top][bottom]vstack=inputs=2:shortest=1[v]")
    _ffmpeg(["-i", str(main_video), "-ss", f"{offset:.2f}", "-stream_loop", "-1", "-i", str(gp),
             "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.2f}",
             "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             str(out_video)])
    log.info("split screen -> %s (%s с %.1fs)", out_video, gp.name, offset)
    return out_video


# ---------------------------------------------------------------------------
# Конвейер
# ---------------------------------------------------------------------------

def _seed_from(payload: dict, device_id: str, source_idx: int) -> int:
    s = payload.get("seed")
    if isinstance(s, int):
        return s & 0x7FFFFFFF
    return int(hashlib.sha256(f"{device_id}|{source_idx}".encode()).hexdigest()[:8], 16)


def _valid_lock(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2 ** 32:
        return value
    return None


def _persona_seed(payload: dict, device_id: str) -> int:
    """Seed лица: стабилен между запусками и не зависит от сцен/дня.
    Порядок: payload.persona_lock → persona_lock master-сцены → sha256(device_id)[:8]."""
    lock = _valid_lock(payload.get("persona_lock"))
    if lock is None:
        for sc in payload.get("scenes") or []:
            if isinstance(sc, dict) and sc.get("role") == "master":
                lock = _valid_lock(sc.get("persona_lock"))
                break
    if lock is None:
        lock = int(hashlib.sha256(device_id.encode()).hexdigest()[:8], 16)
    return lock


def stage_wan(tmp: Path, payload: dict, device_id: str, source_idx: int,
              fetch_master: Callable[[Path], bool] | None,
              store_master: Callable[[Path], None] | None) -> Path:
    scenes: list[dict] = payload["scenes"]
    negative = payload.get("negative_prompt") or DEFAULT_NEGATIVE
    seed = _seed_from(payload, device_id, source_idx)

    master_scenes = [s for s in scenes if s.get("role") == "master"]
    video_scenes = [s for s in scenes if s.get("role") != "master"]
    if not video_scenes:  # старый формат: первая сцена — опорная, но и в видео тоже
        video_scenes = scenes

    master_img = tmp / "master_ref.png"
    have_master = bool(fetch_master and fetch_master(master_img))
    if not have_master:
        prompt = (master_scenes[0] if master_scenes else video_scenes[0])["prompt"]
        # опорный кадр персонажа один на устройство: сид зависит только от persona_lock/device_id
        persona_seed = _persona_seed(payload, device_id)
        log.info("опорный кадр %s: persona_seed=%d", device_id, persona_seed)
        gen_master_image(prompt, negative, master_img, persona_seed)
        if store_master:
            try:
                store_master(master_img)
            except Exception as e:
                log.warning("опорный кадр не сохранён: %s", e)

    clips: list[Path] = []
    for i, sc in enumerate(video_scenes):
        clip = tmp / f"clip_{i:02d}.mp4"
        gen_scene_video(sc["prompt"], negative, master_img,
                        float(sc.get("duration_sec") or 5), clip, seed + i)
        clips.append(clip)
    return concat_clips(clips, tmp / "concat.mp4")


def stage_lipsync(tmp: Path, concat_video: Path, payload: dict, voice_id: str,
                  use_split: bool, seed: int) -> Path:
    narration = (payload.get("narration") or "").strip()
    if not narration:
        narration = " ".join(s.get("narration", "") for s in payload.get("scenes", [])).strip()

    base = concat_video
    if use_split:
        cands = sorted(GAMEPLAY_DIR.glob("*.mp4")) if GAMEPLAY_DIR.exists() else []
        if cands:
            base = split_screen(concat_video, cands, tmp / "split.mp4", seed)
        else:
            log.warning("split_screen включён, но папка gameplay пуста — обычный путь")

    if not narration:  # раньше сюда попадал текст промпта и озвучивался вслух
        log.info("нет текста для озвучки — видео без голоса")
        return base

    audio = synth_voice(narration, voice_id, tmp / "voice.wav")
    final = tmp / "final.mp4"
    if base is concat_video:
        return lipsync(base, audio, final)
    return mux_audio(base, audio, final)  # на split screen губы не синхронизируем


def run_pipeline(payload: dict, device_id: str, source_idx: int, voice_id: str,
                 use_split: bool, tmp: Path,
                 fetch_master: Callable[[Path], bool] | None = None,
                 store_master: Callable[[Path], None] | None = None) -> Path:
    if not payload.get("scenes"):
        raise ValueError("в payload нет scenes")
    with _gpu_lock:  # одна генерация за раз — иначе нехватка видеопамяти
        concat = stage_wan(tmp, payload, device_id, source_idx, fetch_master, store_master)
        return stage_lipsync(tmp, concat, payload, voice_id, use_split,
                             _seed_from(payload, device_id, source_idx))


def _gcs_master_io(payload: dict):
    uri = payload.get("master_ref_uri")
    if not uri:
        return None, None

    def fetch(local: Path) -> bool:
        try:
            if gcs_exists(uri):
                gcs_download(uri, local)
                return True
        except Exception as e:
            log.warning("опорный кадр не скачан (%s): %s", uri, e)
        return False

    def store(local: Path) -> None:
        gcs_upload(local, uri)

    return fetch, store


def run_job() -> dict:
    device_id = os.environ.get("DEVICE_ID", "")
    if not ID_RE.fullmatch(device_id):
        raise ValueError(f"некорректный DEVICE_ID: {device_id!r}")
    try:
        source_idx = int(os.environ.get("SOURCE_IDX", "1"))
    except ValueError as e:
        raise ValueError("SOURCE_IDX должен быть целым числом") from e
    scenes_uri = os.environ.get("SCENES_URI", "")
    output_uri = os.environ.get("OUTPUT_URI", "")
    _parse_gs(scenes_uri)
    _parse_gs(output_uri)
    stage = os.environ.get("STAGE", "full")
    if stage not in STAGES:
        raise ValueError(f"STAGE должен быть одним из {STAGES}, получено {stage!r}")

    with tempfile.TemporaryDirectory(prefix="job_") as tmp_s:
        tmp = Path(tmp_s)
        payload = json.loads(gcs_download(scenes_uri, tmp / "scenes.json").read_text(encoding="utf-8"))
        voice_id = os.environ.get("VOICE_ID") or payload.get("voice_id") or "voice_01"
        use_split = os.environ.get("SPLIT_SCREEN", "1" if payload.get("split_screen") else "0") == "1"
        fetch, store = _gcs_master_io(payload)
        log.info("задача %s/source_%d stage=%s сцен=%d голос=%s split=%s",
                 device_id, source_idx, stage, len(payload.get("scenes", [])), voice_id, use_split)

        concat_uri = os.environ.get("CONCAT_URI") or output_uri.rsplit("/", 1)[0] + "/concat.mp4"
        if stage == "wan":
            with _gpu_lock:
                concat = stage_wan(tmp, payload, device_id, source_idx, fetch, store)
            gcs_upload(concat, concat_uri)
            return {"stage": "wan", "output_uri": concat_uri}

        if stage == "lipsync":
            concat = gcs_download(concat_uri, tmp / "concat.mp4")
            with _gpu_lock:
                final = stage_lipsync(tmp, concat, payload, voice_id, use_split,
                                      _seed_from(payload, device_id, source_idx))
            gcs_upload(final, output_uri)
            return {"stage": "lipsync", "output_uri": output_uri}

        final = run_pipeline(payload, device_id, source_idx, voice_id, use_split, tmp, fetch, store)
        gcs_upload(final, output_uri)
        return {"stage": "full", "device_id": device_id, "source_idx": source_idx,
                "output_uri": output_uri, "scenes": len(payload["scenes"]),
                "split_screen": use_split, "voice_id": voice_id}


# ---------------------------------------------------------------------------
# HTTP-сервер (FastAPI)
# ---------------------------------------------------------------------------

def serve(do_warm: bool = False) -> None:
    import hmac
    from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    token = os.getenv("MODEL_SERVER_TOKEN", "")
    if not token and os.getenv("ALLOW_NO_AUTH") != "1":
        raise SystemExit("задай MODEL_SERVER_TOKEN (или ALLOW_NO_AUTH=1 только для локальной отладки)")

    if do_warm:
        try:
            warm()
        except Exception as e:
            log.warning("warm не удался: %s", e)

    app = FastAPI(title="model_server")

    class ScenesPayload(BaseModel):
        device_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
        scenes: list[dict] = Field(min_length=1, max_length=8)
        voice_id: str = Field(default="voice_01", pattern=r"^[A-Za-z0-9_\-]{1,64}$")
        split_screen: bool = False
        narration: str = Field(default="", max_length=500)
        seed: int | None = None
        negative_prompt: str = Field(default="", max_length=1000)
        persona_lock: int | None = Field(default=None, ge=0, le=2 ** 32 - 1)

    def _auth(authorization: str | None) -> None:
        if not token:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health():
        return {"ok": True, "wan": WAN_DIR.exists(), "xtts": XTTS_DIR.exists(),
                "longcat": LONGCAT_DIR.exists(), "sdxl": SDXL_DIR.exists(),
                "voices": sorted(p.stem for p in VOICES_DIR.glob("*.wav")) if VOICES_DIR.exists() else [],
                "gameplay": len(list(GAMEPLAY_DIR.glob("*.mp4"))) if GAMEPLAY_DIR.exists() else 0,
                "busy": _gpu_lock.locked()}

    @app.post("/generate")
    def generate(p: ScenesPayload, background: BackgroundTasks,
                 authorization: str | None = Header(default=None)):
        _auth(authorization)
        tmp = Path(tempfile.mkdtemp(prefix="srv_"))
        background.add_task(shutil.rmtree, tmp, True)
        payload = p.model_dump()
        masters = MODELS_DIR / "masters"
        # явный persona_lock → отдельный файл кэша, иначе старый кадр перекроет новое лицо
        master_name = (f"{p.device_id}_{p.persona_lock}.png" if p.persona_lock is not None
                       else f"{p.device_id}.png")

        def fetch(local: Path) -> bool:
            src = masters / master_name
            if src.exists():
                shutil.copyfile(src, local)
                return True
            return False

        def store(local: Path) -> None:
            masters.mkdir(parents=True, exist_ok=True)
            tmp_master = masters / f".{master_name}.tmp"
            shutil.copyfile(local, tmp_master)
            os.replace(tmp_master, masters / master_name)

        try:
            final = run_pipeline(payload, p.device_id, 1, p.voice_id, p.split_screen, tmp,
                                 fetch, store)
        except Exception as e:
            log.exception("generate упал")
            raise HTTPException(status_code=500, detail=str(e)[:500])
        return FileResponse(str(final), media_type="video/mp4", filename=f"{p.device_id}.mp4")

    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")  # в контейнере Cloud Run нужен 0.0.0.0, защита — токеном
    uvicorn.run(app, host=host, port=_env_int("PORT", 8080))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve(do_warm="--warm" in sys.argv)
    else:
        if "--warm" in sys.argv:
            warm()
        try:
            sys.stdout.write(json.dumps(run_job(), ensure_ascii=False, indent=2) + "\n")
        except Exception:
            log.exception("задача завершилась с ошибкой")
            sys.exit(1)  # ненулевой код — Cloud Run пометит выполнение как FAILED
