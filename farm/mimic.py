"""mimic.py — планировщик публикаций (N постов в день × plan_days дней).

План устройства лежит в data/post_plans/<device_id>.json (всегда актуальный),
слот дня с номером source_idx соответствует видео source_<idx>.mp4 этого дня.

Разнообразие между аккаунтами одной ниши (когорта = активные устройства с той же niche):
  * хук — пул = статические хуки + шаблоны × подстановки (≈100–200 фраз на нишу и язык).
    На каждый день своя перестановка пула; аккаунт с рангом r в слоте s берёт элемент
    r*S + s. В один день хуки внутри ниши не повторяются, пока N*S <= размера пула
    (иначе предупреждение в лог).
  * хэштеги — все сочетания по 4 и 5 тегов из пула (до hashtag_pool_size=10 тегов),
    перестановка по нише, индекс = (номер_дня*N + r)*S + s. Сочетания не повторяются ни
    между аккаунтами, ни между днями, пока дней*N*S <= числа сочетаний
    (10 тегов → 462 сочетания: хватает на 30 аккаунтов × 2 поста × 7 дней).
  * CTA — 3 варианта на нишу и язык, сдвиг по (ранг, день, слот).
  * реклама (advertising.enabled, устройства с account_type из advertising.account_types):
    cpa_matcher подбирает оффер по хуку и хэштегам слота, link_rotator строит ссылку на каждую
    площадку; в слот пишутся ad и description = «Реклама. <рекламодатель>. erid: …» + ссылка.
    Маркировку площадок и проверку erid делает autoposter (advertising.py).
Всё детерминировано: повторный запуск даёт тот же план.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
import string
import sys
import time
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path
from typing import Any

from config_loader import AppConfig
from devices_registry import DevicesRegistry
from diversity import atomic_write_text, cohort_rank, niche_cohorts, permutation, validate_device_id

log = logging.getLogger(__name__)

NICHE_HOOKS: dict[str, list[str]] = {
    "wrestling_female": ["Ты бы выдержал со мной раунд?", "Смотри, как я уложу тебя на лопатки…",
                         "One round. That's all I need.", "Think you can pin me?", "Ready for the mat?"],
    "yoga_female": ["Медленно… повторяй за мной.", "Твоё тело может больше, чем ты думаешь.",
                    "Can your body do this?", "Slow down… follow me.", "Breathe with me."],
    "fitness_female": ["Один подход — и ты уже не тот.", "Хочешь мою программу?",
                       "Ready for my program?", "One set and you're done.", "No excuses today."],
    "dance_female": ["Повтори это движение, если сможешь.", "Ритм у тебя в голове — или в бёдрах?",
                     "Can you keep up?", "Watch my hips, not my hands.", "Feel the beat."],
    "beach_female": ["Солнце, волны и я — что ещё нужно?", "Поймай эту волну со мной.",
                     "Catch this wave with me?", "Sun, salt, and me.", "Ocean mood on."],
    "home_female": ["Кофе, плед и я — идеальное утро.", "Ты бы остался на завтрак?",
                    "Stay for breakfast?", "Coffee in bed — your move.", "Slow morning with me."],
}
NICHE_CTAS: dict[str, list[str]] = {
    "wrestling_female": ["Follow for round 2", "Tap in 🔥", "Who wins?"],
    "yoga_female": ["Save this flow", "Follow for daily stretch", "Tag a friend"],
    "fitness_female": ["Get the program in bio", "Save this set", "Follow for more"],
    "dance_female": ["Learn it with me", "Save this move", "Follow for part 2"],
    "beach_female": ["Follow for more sun", "Save for summer", "Tag your beach buddy"],
    "home_female": ["Follow for cozy mornings", "Save this vibe", "Stay a while"],
}
NICHE_CTAS_RU: dict[str, list[str]] = {
    "wrestling_female": ["Подпишись на второй раунд", "Кто победит? Пиши 👇", "Сохрани приём"],
    "yoga_female": ["Сохрани этот комплекс", "Подпишись на ежедневную растяжку", "Отправь подруге"],
    "fitness_female": ["Программа — в шапке профиля", "Сохрани эту тренировку", "Подпишись, будет ещё"],
    "dance_female": ["Разучи со мной", "Сохрани связку", "Подпишись на вторую часть"],
    "beach_female": ["Подпишись, будет больше солнца", "Сохрани на лето", "Отметь друга для пляжа"],
    "home_female": ["Подпишись на уютные утра", "Сохрани настроение", "Останься подольше"],
}
GENERIC_HOOKS = ["Look at this.", "Wait for it…", "You won't believe this."]
GENERIC_CTAS = ["Follow for more", "Save this"]
GENERIC_CTAS_RU = ["Подпишись, будет ещё", "Сохрани себе"]

# Шаблоны хуков. Поле {n} — число, остальные поля — подстановки из HOOK_FILLERS.
# Падежи в русских подстановках согласованы с конкретным шаблоном (разные ключи: x, xa, xn …).
HOOK_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "wrestling_female": {
        "en": ["Think you can last {n} seconds in my {x}?", "{n} seconds in my {x}. Ready?",
               "Bet you can't escape my {x}!", "Why does my {x} work every time?",
               "Round {n}: can you stop my {x}?"],
        "ru": ["Продержишься {n} секунд в моём {xp}?", "Раунд {n}: остановишь мой {xa}?",
               "Спорим, не выберешься из моего {xg}?", "Почему мой {xa} срабатывает всегда?",
               "{n} секунд в {xp} — выдержишь?"],
    },
    "yoga_female": {
        "en": ["Can you hold {x} for {n} seconds?", "{n} seconds of {x} — try it with me?",
               "Why is {x} harder than it looks?", "Your hips after {n} days of this?",
               "Follow me: {x}, {n} seconds!"],
        "ru": ["Удержишь {xa} {n} секунд?", "{n} секунд — и спина скажет спасибо!",
               "Почему {xn} сложнее, чем кажется?", "Повторяй за мной: {xn}, {n} секунд!",
               "Что будет с телом за {n} дней растяжки?"],
    },
    "fitness_female": {
        "en": ["Can you do {n} {x} with me?", "{n} {x} a day — what happens?",
               "Bet you quit before {n} {x}!", "Why are {x} better than an hour of cardio?",
               "Only {n} {x}. Ready?"],
        "ru": ["Сделаешь {n} {xg} вместе со мной?", "{n} {xg} в день — что будет через месяц?",
               "Спорим, сдашься раньше, чем на {n} {xp}?", "Почему {xn} лучше часа кардио?",
               "Всего {n} {xg}. Готов?"],
    },
    "dance_female": {
        "en": ["Can you learn {x} in {n} seconds?", "Watch {x} — then try it!",
               "{n} seconds to nail {x}. Go!", "Why does {x} look so easy?",
               "Can you copy {x} on the first try?"],
        "ru": ["Повторишь {xa} за {n} секунд?", "Смотри на {xa} — и повторяй!",
               "Почему {xn} выглядит так легко?", "{n} секунд на {xa}. Погнали!",
               "Получится {xn} с первого раза?"],
    },
    "beach_female": {
        "en": ["{n} minutes of {x} — join me?", "Why is {x} my favorite part of the day?",
               "Who's coming for {x}?", "Day {n}: still chasing {x}!",
               "Would you wake up early for {x}?"],
        "ru": ["{n} минут — и ты забудешь про город!", "Идёшь со мной на {xa}?",
               "День {n}: снова ловлю {xw}!", "Почему {xn} лучше любого кофе?",
               "Встанешь пораньше ради {xg}?", "{n} минут у воды — и ты другой человек!",
               "Сколько раз за {n} минут поймаешь {xw}?", "{n} минут: {xn} и никакого телефона!"],
    },
    "home_female": {
        "en": ["Why does {x} change the whole day?", "Day {n} of {x} — still worth it?",
               "Would you trade your alarm for {x}?", "{n} minutes of {x} — stay with me?",
               "Is {x} the best habit you can start?"],
        "ru": ["Попробуешь {xa} со мной?", "День {n}: {xn} — работает?",
               "Почему {xn} меняет весь день?", "{n} минут уюта — останешься?",
               "Променяешь будильник на {xa}?", "{n} минут: {xn} без телефона — сможешь?",
               "Готов отдать {n} минут на {xa}?"],
    },
}
HOOK_FILLERS: dict[str, dict[str, dict[str, list[str]]]] = {
    "wrestling_female": {
        "en": {"n": ["5", "10", "15", "20", "30", "45", "60"],
               "x": ["guard", "takedown", "armbar", "headlock", "clinch", "sprawl"]},
        "ru": {"n": ["5", "10", "15", "20", "30", "45", "60"],
               "xp": ["захвате", "клинче", "партере", "замке"],
               "xa": ["бросок", "проход в ноги", "рычаг", "удушающий"],
               "xg": ["захвата", "клинча", "замка", "удержания"]},
    },
    "yoga_female": {
        "en": {"n": ["10", "15", "20", "30", "45", "60"],
               "x": ["tree pose", "crow pose", "warrior three", "a deep squat", "boat pose",
                     "side plank"]},
        "ru": {"n": ["10", "15", "20", "30", "45", "60"],
               "xa": ["позу дерева", "позу вороны", "позу лодки", "боковую планку",
                      "глубокий присед"],
               "xn": ["поза дерева", "поза вороны", "поза лодки", "боковая планка",
                      "глубокий присед"]},
    },
    "fitness_female": {
        "en": {"n": ["10", "12", "15", "20", "25", "30", "50"],
               "x": ["squats", "push-ups", "lunges", "burpees", "jump squats",
                     "mountain climbers"]},
        "ru": {"n": ["10", "12", "15", "20", "25", "30", "50"],
               "xg": ["приседаний", "отжиманий", "выпадов", "бёрпи", "скручиваний"],
               "xp": ["приседаниях", "отжиманиях", "выпадах", "скручиваниях"],
               "xn": ["приседания", "отжимания", "выпады", "бёрпи", "скручивания"]},
    },
    "dance_female": {
        "en": {"n": ["10", "15", "20", "30", "45", "60"],
               "x": ["this hip roll", "this footwork", "the body wave", "this spin",
                     "this shuffle", "this arm combo"]},
        "ru": {"n": ["10", "15", "20", "30", "45", "60"],
               "xa": ["эту волну", "этот шаффл", "эту связку", "этот поворот", "эту дорожку"],
               "xn": ["эта волна", "этот шаффл", "эта связка", "этот поворот", "эта дорожка"]},
    },
    "beach_female": {
        "en": {"n": ["5", "10", "15", "20", "30", "45"],
               "x": ["sunrise surf", "a sunset swim", "low tide", "a beach run", "a paddle out",
                     "golden hour"]},
        "ru": {"n": ["5", "10", "15", "20", "30", "45"],
               "xa": ["рассвет", "закат", "пробежку", "заплыв", "сёрф"],
               "xw": ["волну", "закат", "рассвет", "ветер"],
               "xn": ["океан", "утренний заплыв", "закат у воды", "солёный ветер"],
               "xg": ["рассвета", "волны", "заплыва", "тишины у воды"]},
    },
    "home_female": {
        "en": {"n": ["3", "5", "7", "10", "14", "21", "30"],
               "x": ["slow mornings", "a coffee ritual", "a cozy reset", "Sunday pancakes",
                     "a five-minute tidy", "rainy-day reading"]},
        "ru": {"n": ["5", "7", "10", "14", "20", "30"],
               "xa": ["медленное утро", "кофейный ритуал", "уютную перезагрузку",
                      "воскресные блинчики"],
               "xn": ["медленное утро", "кофейный ритуал", "уютная перезагрузка",
                      "утренний плед"]},
    },
}

TAGS_PER_POST = 4          # оставлено для совместимости; теперь 4 или 5 (TAG_COUNTS)
TAG_COUNTS = (4, 5)
HASHTAG_POOL_SIZE = 10
HOOK_MAX_LEN = 90
_CYR = re.compile(r"[А-Яа-яЁё]")
_WARNED: set[tuple] = set()  # предупреждения о нехватке пула — один раз на нишу, а не на устройство


def match_tone(niche: str) -> tuple[list[str], list[str]]:
    """Совместимость: статические хуки и CTA ниши (без шаблонов)."""
    return NICHE_HOOKS.get(niche, GENERIC_HOOKS), NICHE_CTAS.get(niche, GENERIC_CTAS)


def _lang(cfg: AppConfig) -> str:
    lang = str(getattr(getattr(cfg, "forge", None), "language", "en") or "en").lower()
    return lang if lang in ("en", "ru") else "en"


def _is_lang(text: str, lang: str) -> bool:
    return bool(_CYR.search(text)) == (lang == "ru")


def _expand_template(tpl: str, fills: dict[str, list[str]]) -> list[str]:
    fields = list(dict.fromkeys(f for _, f, _, _ in string.Formatter().parse(tpl) if f))
    if not fields:
        return [tpl]
    missing = [f for f in fields if not fills.get(f)]
    if missing:
        log.warning("шаблон хука %r: нет подстановок для %s", tpl, missing)
        return []
    return [tpl.format(**dict(zip(fields, combo)))
            for combo in itertools.product(*(fills[f] for f in fields))]


def hook_pool(cfg: AppConfig, niche: str, lang: str | None = None) -> list[str]:
    """Все хуки ниши на языке lang, отсортированы (порядок не зависит от словарей)."""
    lang = lang or _lang(cfg)
    static = [h for h in NICHE_HOOKS.get(niche, []) if _is_lang(h, lang)]
    extra = [str(h) for h in (cfg.niches.get(niche, {}) or {}).get("hooks", []) or []
             if _is_lang(str(h), lang)]  # niches.json → <niche>.hooks, если задано
    templated: list[str] = []
    for tpl in HOOK_TEMPLATES.get(niche, {}).get(lang, []):
        templated += _expand_template(tpl, HOOK_FILLERS.get(niche, {}).get(lang, {}))
    pool = [h.strip() for h in static + extra + templated]
    pool = sorted(dict.fromkeys(h for h in pool if h and len(h) <= HOOK_MAX_LEN))
    return pool or list(GENERIC_HOOKS)


def cta_pool(niche: str, lang: str) -> list[str]:
    if lang == "ru":
        return NICHE_CTAS_RU.get(niche, GENERIC_CTAS_RU)
    return NICHE_CTAS.get(niche, GENERIC_CTAS)


def _load_hashtags(cfg: AppConfig, niche: str) -> list[str]:
    tags: list[str] = []
    p = cfg.path("hashtag_seeds")
    if p.exists():
        try:
            tags = list(json.loads(p.read_text(encoding="utf-8")).get(niche, []))
        except Exception as e:
            log.warning("hashtag_seeds.json не читается: %s", e)
    if not tags:
        tags = list(cfg.niches.get(niche, {}).get("hashtags_seed", []))
    # только валидные уникальные теги
    clean: list[str] = []
    for t in tags:
        t = str(t).strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = "#" + t
        if t.lower() not in (x.lower() for x in clean):
            clean.append(t)
    return clean[:15]


def hashtag_combos(pool: list[str], sizes: tuple[int, ...] = TAG_COUNTS) -> list[tuple[str, ...]]:
    combos = [c for k in sizes if 0 < k <= len(pool) for c in itertools.combinations(pool, k)]
    if not combos and pool:  # тегов меньше 4 — один набор на всех
        combos = [tuple(pool)]
    return combos


def plan_file(cfg: AppConfig, device_id: str) -> Path:
    validate_device_id(device_id)
    d = cfg.path("post_plans")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{device_id}.json"


def _pick_no_repeat(pool: list[str], last: str | None, rnd) -> str:
    """Совместимость (больше не используется в build_plans)."""
    choices = [x for x in pool if x != last] or pool
    return rnd.choice(choices)


def build_plans(cfg: AppConfig, device: dict, plan_days: int, posts_per_day: int,
                start: date_cls | None = None, rnd: Any = None,
                cohort: list[str] | None = None, ad_planner: Any = None) -> dict[str, Any]:
    """cohort — device_id всех активных устройств той же ниши (для уникальности между ними).
    rnd оставлен для совместимости и не используется: выбор детерминирован."""
    device_id = validate_device_id(device["device_id"])
    niche = device["niche"]
    lang = _lang(cfg)
    hooks = hook_pool(cfg, niche, lang)
    ctas = cta_pool(niche, lang)
    pool_size = max(1, int(cfg.mimic.get("hashtag_pool_size", HASHTAG_POOL_SIZE)))
    combos = hashtag_combos(_load_hashtags(cfg, niche)[:pool_size])
    tag_perm = permutation(len(combos), "tags", niche)
    rank, n = cohort_rank(cohort or [device_id], device_id)
    per_day = max(1, int(posts_per_day))

    warn_key = (niche, lang, n, per_day, plan_days)
    first_warning = warn_key not in _WARNED
    _WARNED.add(warn_key)
    if first_warning and n * per_day > len(hooks):
        log.warning("mimic %s: хуков в пуле %d < %d (аккаунтов %d × слотов %d) — "
                    "в один день хуки будут повторяться; добавь шаблоны или niches.json hooks",
                    niche, len(hooks), n * per_day, n, per_day)
    if first_warning and combos and plan_days * n * per_day > len(combos):
        log.warning("mimic %s: сочетаний хэштегов %d < %d — будут повторы; увеличь "
                    "mimic.hashtag_pool_size или пул тегов", niche, len(combos),
                    plan_days * n * per_day)

    platforms = [p for p, acc in (device.get("accounts") or {}).items() if acc.get("enabled", True)]
    platforms = [p for p in platforms if p in cfg.uniqueness.get("platforms", platforms)]
    plan: dict[str, Any] = {"device_id": device_id, "niche": niche,
                            "marker": device.get("marker", ""), "platforms": platforms,
                            "lang": lang, "cohort_rank": rank, "cohort_size": n,
                            "created_at": time.time(), "days": []}
    start = start or date_cls.fromisoformat(cfg.today())
    for d_off in range(plan_days):
        day_date = start + timedelta(days=d_off)
        day = day_date.isoformat()
        hook_perm = permutation(len(hooks), "hooks", niche, lang, day)
        slots = []
        for idx in range(1, per_day + 1):
            g = rank * per_day + (idx - 1)
            hook = hooks[hook_perm[g % len(hooks)]]
            cta = ctas[(rank + day_date.toordinal() + idx) % len(ctas)]
            tags: list[str] = []
            if combos:
                ti = ((day_date.toordinal() * n + rank) * per_day + (idx - 1)) % len(combos)
                tags = list(combos[tag_perm[ti]])
            slot: dict[str, Any] = {"slot": idx, "source_idx": idx, "status": "planned",
                                    "hook": hook, "cta": cta, "hashtags": tags, "platforms": platforms}
            if ad_planner is not None:
                ad = ad_planner.ad_for(device, platforms, hook, tags)
                if ad is not None:
                    slot["ad"] = ad
                    slot["description"] = {p: f"{ad['label']}\n{ad['links'][p]}" for p in platforms}
            slots.append(slot)
        plan["days"].append({"date": day, "slots": slots})
    return plan


def load_plan(cfg: AppConfig, device_id: str) -> dict:
    f = plan_file(cfg, device_id)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("план %s не читается: %s", f, e)
        return {}


def get_slot(cfg: AppConfig, device_id: str, day: str, source_idx: int) -> dict:
    """Слот плана для устройства на дату day с номером source_idx (или {})."""
    for d in load_plan(cfg, device_id).get("days", []):
        if d.get("date") != day:
            continue
        for slot in d.get("slots", []):
            if int(slot.get("source_idx", -1)) == int(source_idx):
                return slot
    return {}


def _write_atomic(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


class Mimic:
    def __init__(self, cfg: AppConfig, registry: DevicesRegistry | None = None, health: Any = None,
                 ad_planner: Any = None):
        """health — AccountHealth; по умолчанию создаётся, если account_health.enabled=true.
        ad_planner — advertising.AdPlanner; по умолчанию создаётся, если advertising.enabled=true."""
        self.cfg = cfg
        self.registry = registry or DevicesRegistry.from_config(cfg)
        if health is None:
            from account_health import health_for
            health = health_for(cfg)
        self.health = health
        if ad_planner is None:
            from advertising import AdPlanner
            ad_planner = AdPlanner.from_config(cfg)
        self.ad_planner = ad_planner
        self.plan_days = max(1, int(cfg.mimic.get("plan_days", 7)))
        self.posts_per_day = max(1, int(cfg.mimic.get("posts_per_account_per_day",
                                                      cfg.forge.videos_per_device_per_day)))

    def run(self, device_ids: list[str] | None = None, replan: bool = False) -> dict[str, Any]:
        """replan=True — пересобрать слоты со статусом "planned" (нужно один раз после
        обновления, чтобы старые планы получили новые хуки/теги). Без него уже
        запланированные слоты сохраняются как были."""
        devices = self.registry.get_active_devices(device_ids=device_ids)
        if not devices:
            log.warning("mimic: нет активных устройств")
            return {}
        # когорты считаем по ВСЕМ активным устройствам, иначе ранги зависят от --device
        cohorts = niche_cohorts(self.registry.get_active_devices(device_ids=None))
        out: dict[str, Any] = {}
        for device in devices:
            device_id = validate_device_id(device["device_id"])
            if self.health is not None and self.health.enabled:
                allowed = set(self.health.plan_platforms(device))
                skipped = sorted(p for p, a in (device.get("accounts") or {}).items()
                                 if a.get("enabled", True) and p not in allowed)
                if skipped:
                    log.warning("mimic %s: без плана (banned/shadowban_suspect): %s",
                                device_id, skipped)
                if not allowed:
                    out[device_id] = {"skipped": "все аккаунты banned или shadowban_suspect"}
                    continue
                device = {**device, "accounts": {p: a for p, a in (device.get("accounts") or {}).items()
                                                 if p in allowed}}
            plan = build_plans(self.cfg, device, self.plan_days, self.posts_per_day,
                               cohort=cohorts.get(device["niche"], [device_id]),
                               ad_planner=self.ad_planner)
            f = plan_file(self.cfg, device_id)
            old = load_plan(self.cfg, device_id)
            old_slots = {(d["date"], int(s["source_idx"])): s
                         for d in old.get("days", []) for s in d.get("slots", [])
                         if not (replan and s.get("status", "planned") == "planned")}
            for d in plan["days"]:
                d["slots"] = [old_slots.get((d["date"], int(s["source_idx"])), s)
                              for s in d["slots"]]
            _write_atomic(f, plan)
            ads = sum(1 for d in plan["days"] for s in d["slots"] if s.get("ad"))
            out[device_id] = {"plan_file": str(f), "days": len(plan["days"]), "ad_slots": ads,
                              "slots_per_day": self.posts_per_day,
                              "cohort_rank": plan["cohort_rank"], "cohort_size": plan["cohort_size"]}
            log.info("mimic %s: %d дней × %d слотов → %s", device_id,
                     len(plan["days"]), self.posts_per_day, f)
        return out


if __name__ == "__main__":
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--replan", action="store_true",
                        help="пересобрать ещё не опубликованные слоты (status=planned)")

    cfg, a = cli_config("mimic", extra=extra)
    sys.stdout.write(json.dumps(Mimic(cfg).run(a.device, replan=a.replan),
                                ensure_ascii=False, indent=2) + "\n")
