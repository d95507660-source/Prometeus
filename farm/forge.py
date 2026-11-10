"""forge.py — сценарий + сцены + генерация видео через GCP Cloud Run Job.

Результат: data/videos/<device_id>/<YYYY-MM-DD>/source_<idx>.mp4
Контрольные точки: data/forge_checkpoints/<device_id>/<YYYY-MM-DD>/source_<idx>.json

Разнообразие между аккаунтами одной ниши:
  * темы — у каждого аккаунта своя непересекающаяся порция (ранг в когорте ниши).
    Если тем меньше, чем нужно когорте, пул расширяется: при наличии TOGETHER_API_KEY —
    переформулировками от LLM, затем детерминированно «тема × угол подачи»;
  * сценарий — в промпт LLM передаётся persona_angle (тон + внешность из metadata.json,
    либо детерминированные значения по device_id);
  * сцены — ракурс, план (beat) и свет выбираются с сидом от (device_id, day, idx);
  * лицо — если в metadata.json нет appearance, внешность собирается из признаков по
    hash(device_id); seed лица (persona_lock) уходит в master-сцену для model_server.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config_loader import AppConfig
from devices_registry import DevicesRegistry
from diversity import (atomic_write_text, cohort_rank, niche_cohorts, stable_int,
                       validate_day, validate_device_id)

log = logging.getLogger(__name__)

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

# Запасные ключевые слова, если в niches.json их нет
NICHE_KEYWORDS: dict[str, list[str]] = {
    "wrestling_female": ["wrestling", "mma", "grappl", "fight", "ring", "combat", "submission", "борьб"],
    "yoga_female": ["yoga", "stretch", "flexib", "flow", "meditat", "asana", "zen", "йог"],
    "fitness_female": ["fitness", "gym", "lift", "workout", "strength", "cardio", "training", "фитнес"],
    "dance_female": ["dance", "choreo", "moves", "rhythm", "groove", "twerk", "hiphop", "танц"],
    "beach_female": ["beach", "surf", "ocean", "sand", "wave", "swim", "coast", "пляж"],
    "home_female": ["home", "cozy", "morning", "coffee", "pajama", "bedroom", "relax", "уют"],
}

MASTER_REFERENCE_PROMPT = (
    "master reference portrait of an adult athletic woman in her twenties, {appearance}, {niche_style}, "
    "natural skin texture, sharp focus, 85mm lens, soft studio light, neutral background"
)
SCENE_TEMPLATE = (
    "{action}, {beat}, {setting}, {lighting}, camera {camera}, "
    "same face and body as master reference, natural motion, vertical 9:16 framing"
)
NEGATIVE_PROMPT = (
    "extra fingers, deformed hands, distorted face, mutated, disfigured, bad anatomy, "
    "plastic skin, waxy, cartoon, anime, 3d render, text, watermark, logo, blurry, "
    "child, minor, teen"
)
NICHE_STYLE = {
    "wrestling_female": "athletic build, sports bra, fight shorts, confident gaze",
    "yoga_female": "lean flexible build, yoga leggings, calm expression",
    "fitness_female": "toned muscular build, gym wear, determined gaze",
    "dance_female": "expressive dancer build, crop top, playful smile",
    "beach_female": "sun-kissed skin, swimwear, relaxed smile",
    "home_female": "soft cozy build, oversized sweater, gentle smile",
}
CAMERAS = ["slow push in", "handheld follow", "static medium shot", "slight orbit", "low angle",
           "slow pull out", "gentle tilt up"]
SCENE_BEATS = ["wide establishing view", "medium shot", "close-up on face and expression",
               "detail shot of hands and movement", "over-the-shoulder view", "full-body shot"]
LIGHTINGS = ["golden hour light", "soft overcast light", "bright clean daylight",
             "warm tungsten light", "cool blue-hour light", "soft window light",
             "colorful neon accent light"]
PERSONA_TONES = ["playful and teasing", "calm, encouraging coach", "competitive and bold",
                 "dry deadpan humor", "detail-loving explainer", "warm big-sister energy",
                 "high-energy hype", "minimalist, few words"]
# Признаки внешности для аккаунтов без metadata.json → appearance
APPEARANCE_TRAITS: dict[str, list[str]] = {
    "hair_color": ["jet-black hair", "dark brown hair", "chestnut hair", "auburn hair",
                   "copper red hair", "honey blonde hair", "platinum blonde hair", "ash brown hair"],
    "hairstyle": ["long and straight", "shoulder-length waves", "in a high ponytail",
                  "in a messy bun", "in a sleek bob", "in long braids", "naturally curly",
                  "in a short pixie cut"],
    "eyes": ["brown eyes", "hazel eyes", "green eyes", "blue eyes", "grey eyes", "amber eyes"],
    "skin": ["fair skin", "light olive skin", "golden tan skin", "warm beige skin",
             "medium brown skin", "deep brown skin"],
    "face": ["oval face", "heart-shaped face", "round face", "defined square jaw",
             "high cheekbones", "soft angular face"],
    "feature": ["light freckles", "a small beauty mark", "dimples", "full straight eyebrows",
                "a slightly upturned nose", "a wide bright smile"],
}
TOPIC_ANGLES = ["beginner mistake", "myth vs fact", "behind the scenes", "day in the life",
                "30-second challenge", "before and after", "one quick tip",
                "what nobody tells you", "answering a common question", "step-by-step breakdown",
                "first try vs 100th try", "rating it honestly"]
HEADERS = {"User-Agent": "ai-girls-studio/1.1", "Content-Type": "application/json"}

HOOK_MAX_LEN = 90
TOPICS_PER_DEVICE = 3
MAX_LLM_TOPICS = 60


@dataclass
class HookResult:
    hook: str
    score: int
    attempts: int
    ok: bool


@dataclass
class Scene:
    prompt: str
    duration_sec: int
    reference: str | None = None
    role: str = "scene"  # "master" — только опорный кадр, в итоговое видео не попадает
    persona_lock: int | None = None  # seed лица, только у master-сцены

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"prompt": self.prompt, "duration_sec": self.duration_sec,
                             "reference": self.reference, "role": self.role}
        if self.persona_lock is not None:
            d["persona_lock"] = self.persona_lock
        return d


@dataclass
class ForgeResult:
    device_id: str
    source_idx: int
    day: str
    local_path: str | None
    hook: HookResult
    script: dict = field(default_factory=dict)
    scenes: list[Scene] = field(default_factory=list)
    realism_score: int = 0
    status: str = "unknown"
    error: str | None = None
    gcp_execution_name: str | None = None  # имя выполнения Cloud Run (для очереди и отмены)
    gpu_seconds: float | None = None       # длительность выполнения Cloud Run (для budget_guard)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Оценка хука
# ---------------------------------------------------------------------------

_W_YOU = re.compile(r"\b(ты|тебя|тебе|твой|твоя|твоё|you|your|you're|how|why|what)\b", re.I)
_W_STRONG = re.compile(r"\b(never|always|secret|only|никогда|секрет|всегда|только)\b", re.I)
_W_LOOK = re.compile(r"\b(look|watch|wait|смотри|стой|глянь|жди)\b", re.I)


def score_hook(hook: str) -> int:
    hook = (hook or "").strip()
    if not hook:
        return 0
    score = 0
    if 5 <= len(hook) <= 60:
        score += 20
    elif len(hook) <= HOOK_MAX_LEN:
        score += 10
    if any(c in hook for c in "?!"):
        score += 15
    if re.search(r"\d", hook):
        score += 15
    if _W_YOU.search(hook):
        score += 20
    if _W_STRONG.search(hook):
        score += 15
    if _W_LOOK.search(hook):
        score += 15
    return min(score, 100)


def _clean_text(text: str, limit: int = 200) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"[#@]\S+", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _clean_hook(text: str) -> str:
    return _clean_text(str(text or ""), HOOK_MAX_LEN).strip(" \"'«»")


def _niche_keywords(cfg: AppConfig, niche: str) -> list[str]:
    kws = [k.lower() for k in cfg.niches.get(niche, {}).get("keywords", [])]
    return list(dict.fromkeys(kws + NICHE_KEYWORDS.get(niche, [])))


def _niche_matches(text: str, kws: list[str]) -> bool:
    if not kws:
        return True
    low = (text or "").lower()
    return any(k in low for k in kws)


class RealismChecker:
    """Быстрая проверка промптов перед дорогой генерацией (не оценка готового видео)."""
    BAD_TOKENS = [
        "extra finger", "deformed", "distorted", "mutated", "disfigured", "bad anatomy",
        "wrong anatomy", "fused fingers", "plastic skin", "waxy", "uncanny", "cartoon",
        "anime", "3d render",
    ]

    def __init__(self, min_score: int = 70):
        self.min_score = min_score

    def score_scene(self, scene: dict) -> int:
        prompt = (scene.get("prompt") or "").lower()
        score = 100
        for bad in self.BAD_TOKENS:
            if bad in prompt:
                score -= 25
        if scene.get("role") == "master":
            if "master reference" not in prompt:
                score -= 30
        else:
            if "master reference" not in prompt:
                score -= 15
            if not scene.get("reference"):
                score -= 20
            if (scene.get("duration_sec") or 0) < 3:
                score -= 10
        return max(0, score)

    def check(self, scenes: list[dict]) -> tuple[bool, int]:
        if not scenes or not any(s.get("role") != "master" for s in scenes):
            return False, 0
        scores = [self.score_scene(s) for s in scenes]
        avg = int(sum(scores) / len(scores))
        return avg >= self.min_score, avg


# ---------------------------------------------------------------------------
# Тренды
# ---------------------------------------------------------------------------

def _read_items(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("не читается %s: %s", path, e)
        return []
    if isinstance(data, dict):
        data = data.get("items", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _load_trend_analyzer_trends(cfg: AppConfig, niche: str) -> list[dict]:
    f = cfg.path("trend_analysis") / f"{niche}.json"
    return _read_items(f) if f.exists() else []


def _load_trends_cache(cfg: AppConfig, niche: str) -> list[dict]:
    cache_dir = cfg.path("trends_cache")
    if not cache_dir.exists():
        return []
    out: list[dict] = []
    for f in sorted(cache_dir.glob(f"{niche}_*.json")):
        out.extend(_read_items(f))
    return out


# ---------------------------------------------------------------------------
# LLM (Together, OpenAI-совместимый chat/completions)
# ---------------------------------------------------------------------------

def _together_chat(api_key: str, model: str, prompt: str, max_tokens: int = 300,
                   temperature: float = 0.9) -> str:
    import requests
    r = requests.post(
        TOGETHER_URL,
        headers={"Authorization": f"Bearer {api_key}", **HEADERS},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=(10, 90),
    )
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def _extract_json(text: str) -> Any:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("в ответе нет JSON")
    return json.loads(m.group(0))


def _llm_topic_variations(niche: str, seeds: list[str], count: int, api_key: str,
                          model: str) -> list[str]:
    """Переформулировки трендов с разными углами подачи (не копии)."""
    count = max(0, min(count, MAX_LLM_TOPICS))
    if not count or not seeds:
        return []
    seed_lines = "\n".join(f"- {s}" for s in seeds[:10])
    prompt = (
        f"You plan short vertical videos for the '{niche}' niche. The character is an adult woman.\n"
        f"Current trends:\n{seed_lines}\n"
        f"Write {count} NEW video topic ideas. Each must take a different angle on these trends "
        "(tutorial, myth-busting, challenge, behind the scenes, story, comparison, reaction...). "
        "Do not copy the trends verbatim. Keep it safe for work. Each idea max 100 characters, "
        "in English.\n"
        'Return STRICT JSON only, no markdown: {"topics": ["idea 1", "idea 2"]}'
    )
    try:
        data = _extract_json(_together_chat(api_key, model, prompt,
                                            max_tokens=min(4000, 60 * count), temperature=1.0))
        items = data.get("topics", []) if isinstance(data, dict) else []
        out = [_clean_text(str(x), 120) for x in items if isinstance(x, (str, int, float))]
        return [x for x in dict.fromkeys(out) if len(x) >= 8][:count]
    except Exception as e:
        log.warning("llm_topic_variations %s не удался: %s", niche, e)
        return []


# ---------------------------------------------------------------------------
# Сценарий
# ---------------------------------------------------------------------------

FALLBACK_ACTIONS = {
    "wrestling_female": [("sparring in the ring", "MMA gym"), ("drilling a takedown on the mat", "wrestling room"),
                         ("shadow boxing between rounds", "boxing gym"), ("wrapping her hands before training", "locker room"),
                         ("practicing sprawls with a partner", "grappling academy")],
    "yoga_female": [("flowing through a stretch sequence", "sunlit yoga studio"), ("holding a balance pose", "rooftop at sunrise"),
                    ("doing a slow sun salutation", "garden terrace"), ("breathing in a seated twist", "quiet beach at dawn"),
                    ("easing into a deep hip opener", "minimal home studio")],
    "fitness_female": [("finishing a heavy set", "modern gym"), ("doing explosive box jumps", "functional training zone"),
                       ("pushing a weighted sled", "indoor turf track"), ("doing kettlebell swings", "garage gym"),
                       ("sprinting short intervals", "outdoor running track")],
    "dance_female": [("hitting a short choreography", "dance studio"), ("freestyling to the beat", "neon-lit room"),
                     ("practicing a spin sequence", "rooftop at dusk"), ("teaching a simple footwork pattern", "bright loft"),
                     ("dancing through a city crosswalk", "quiet street at night")],
    "beach_female": [("walking along the shoreline", "sunset beach"), ("carrying a surfboard into the waves", "surf beach"),
                     ("jogging on wet sand", "early morning beach"), ("paddling out on a board", "calm bay"),
                     ("shaking sand off a towel", "dunes by the sea")],
    "home_female": [("sipping coffee in bed", "cozy bedroom"), ("making breakfast", "bright kitchen"),
                    ("watering plants by the window", "sunny living room"), ("reading under a blanket", "rainy reading nook"),
                    ("tidying a small desk", "calm home office")],
}
FALLBACK_HOOKS = {
    "wrestling_female": ["Think you can pin me?", "One round. That's all I need."],
    "yoga_female": ["Can your body do this?", "Slow down… follow me."],
    "fitness_female": ["Ready for my program?", "One set and you're done."],
    "dance_female": ["Can you keep up?", "Watch my hips, not my hands."],
    "beach_female": ["Catch this wave with me?", "Sun, salt, and me."],
    "home_female": ["Stay for breakfast?", "Coffee in bed — your move."],
}


def _fallback_script(niche: str, topics: list[str], rnd: random.Random | None = None) -> dict:
    rnd = rnd or random.Random()
    action, setting = rnd.choice(FALLBACK_ACTIONS.get(niche, [("standing confidently", "studio")]))
    hooks = FALLBACK_HOOKS.get(niche) or ["Look at this."]
    return {"hook": rnd.choice(hooks), "action": action, "setting": setting,
            "hashtags": ["#" + niche.replace("_female", ""), "#fyp"], "source": "fallback"}


def _normalize_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = tags.split()
    if not isinstance(tags, list):
        return []
    out = []
    for t in tags:
        t = re.sub(r"[^\w]", "", str(t))
        if 2 <= len(t) <= 30:
            out.append("#" + t.lower())
    return list(dict.fromkeys(out))[:6]


def _llm_script(niche: str, topic: str, api_key: str | None, model: str,
                language: str = "en", rnd: random.Random | None = None,
                persona_angle: str = "") -> dict:
    if not api_key:
        return _fallback_script(niche, [topic], rnd)
    try:
        lang_name = {"en": "English", "ru": "Russian"}.get(language, language)
        persona_line = (f"Account persona (write in this voice so this channel sounds unlike "
                        f"others in the niche): {persona_angle}\n") if persona_angle else ""
        prompt = (
            f"You are a TikTok scriptwriter for the '{niche}' niche. The character is an adult woman.\n"
            f"{persona_line}"
            f"Trend inspiration (do not copy it): {topic}\n"
            f"Write the hook in {lang_name}. Keep it safe for work.\n"
            "Return STRICT JSON only, no markdown:\n"
            '{"hook": "short hook, max 60 chars, with ? or !", '
            '"action": "what she does in 5 seconds, in English", '
            '"setting": "location, in English", '
            '"hashtags": ["#tag1", "#tag2", "#tag3"]}'
        )
        data = _extract_json(_together_chat(api_key, model, prompt))
        if not isinstance(data, dict):
            raise ValueError("JSON не объект")
        script = {
            "hook": _clean_hook(data.get("hook", "")),
            "action": _clean_text(str(data.get("action", "")), 160),
            "setting": _clean_text(str(data.get("setting", "")), 80),
            "hashtags": _normalize_tags(data.get("hashtags", [])),
            "source": "llm",
            "topic": topic,
        }
        if not script["action"] or not script["setting"]:
            fb = _fallback_script(niche, [topic], rnd)
            script["action"] = script["action"] or fb["action"]
            script["setting"] = script["setting"] or fb["setting"]
        return script
    except Exception as e:
        log.warning("llm_script не удался: %s", e)
        return _fallback_script(niche, [topic], rnd)


# ---------------------------------------------------------------------------
# Персона
# ---------------------------------------------------------------------------

def _stable_int(*parts: Any) -> int:
    return stable_int(*parts)


def _float_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def _stable_voice_id(device_id: str, pool: list[str]) -> str:
    if not pool:
        return "voice_01"
    return pool[_stable_int(device_id) % len(pool)]


def _persona(cfg: AppConfig, device_id: str) -> dict:
    f = cfg.path("models") / validate_device_id(device_id) / "metadata.json"
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("metadata %s не читается: %s", f, e)
        return {}


def auto_appearance(device_id: str) -> str:
    """Внешность по device_id: одинаковая между запусками, разная между аккаунтами
    (8·8·6·6·6·6 = 82 944 сочетаний; совпадение текста не делает лица одинаковыми —
    seed лица всё равно свой, см. persona_lock)."""
    t = {k: v[_stable_int(device_id, "appearance", k) % len(v)] for k, v in APPEARANCE_TRAITS.items()}
    return (f"{t['hair_color']} {t['hairstyle']}, {t['eyes']}, {t['skin']}, "
            f"{t['face']}, {t['feature']}")


def persona_lock(cfg: AppConfig, device_id: str, persona: dict | None = None) -> int:
    """Seed лица. Из metadata.json → persona_lock, иначе sha256(device_id)[:8] —
    та же формула, что раньше была в model_server, поэтому существующие лица не меняются."""
    persona = _persona(cfg, device_id) if persona is None else persona
    lock = persona.get("persona_lock")
    if isinstance(lock, int) and not isinstance(lock, bool) and 0 <= lock < 2 ** 32:
        return lock
    if lock is not None:
        log.warning("%s: persona_lock=%r вне диапазона 0..2^32-1, использую hash", device_id, lock)
    return _stable_int(device_id)


def persona_angle(cfg: AppConfig, device: dict) -> str:
    device_id = device["device_id"]
    persona = _persona(cfg, device_id)
    tone = _clean_text(str(persona.get("tone") or ""), 80) or \
        PERSONA_TONES[_stable_int(device_id, "tone") % len(PERSONA_TONES)]
    look = _clean_text(str(persona.get("appearance") or ""), 160) or auto_appearance(device_id)
    return f"tone: {tone}; look: {look}"


# ---------------------------------------------------------------------------
# Сцены
# ---------------------------------------------------------------------------

def _build_scenes(cfg: AppConfig, device: dict, script: dict, rnd: random.Random) -> list[Scene]:
    """rnd должен быть засеян от (device_id, day, idx) — см. generate_one."""
    niche = device["niche"]
    device_id = device["device_id"]
    persona = _persona(cfg, device_id)
    appearance = _clean_text(str(persona.get("appearance") or ""), 200) or auto_appearance(device_id)
    style = NICHE_STYLE.get(niche, "athletic build")
    # свет — фирменный для аккаунта, ракурс и план меняются от ролика к ролику
    lighting = _clean_text(str(persona.get("lighting") or ""), 60) or \
        LIGHTINGS[_stable_int(device_id, "lighting") % len(LIGHTINGS)]
    master_ref = MASTER_REFERENCE_PROMPT.format(appearance=appearance, niche_style=style)
    lo, hi = cfg.forge.scenes_per_video
    max_by_duration = max(1, max(cfg.forge.video_duration_sec) // max(1, cfg.forge.scene_duration_sec))
    n_scenes = rnd.randint(lo, max(lo, min(hi, max_by_duration)))
    out = [Scene(prompt=master_ref + ", full body, neutral pose", duration_sec=0,
                 reference=None, role="master",
                 persona_lock=persona_lock(cfg, device_id, persona))]
    cams = rnd.sample(CAMERAS, k=min(n_scenes, len(CAMERAS)))
    beats = rnd.sample(SCENE_BEATS, k=min(n_scenes, len(SCENE_BEATS)))
    for i in range(n_scenes):
        prompt = SCENE_TEMPLATE.format(action=script.get("action") or "moving naturally",
                                       beat=beats[i % len(beats)],
                                       setting=script.get("setting") or "studio",
                                       lighting=lighting,
                                       camera=cams[i % len(cams)])
        out.append(Scene(prompt=prompt, duration_sec=cfg.forge.scene_duration_sec,
                         reference="master_reference"))
    return out


# ---------------------------------------------------------------------------
# Контрольные точки
# ---------------------------------------------------------------------------

def _checkpoint_path(cfg: AppConfig, device_id: str, day: str, source_idx: int) -> Path:
    d = cfg.path("forge_checkpoints") / validate_device_id(device_id) / validate_day(day)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"source_{int(source_idx)}.json"


def read_checkpoint(cfg: AppConfig, device_id: str, day: str, source_idx: int) -> dict:
    p = (cfg.path("forge_checkpoints") / validate_device_id(device_id) / validate_day(day)
         / f"source_{int(source_idx)}.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_done(cfg: AppConfig, device_id: str, day: str, source_idx: int) -> dict | None:
    data = read_checkpoint(cfg, device_id, day, source_idx)
    if data.get("status") == "ok" and data.get("local_path") and Path(data["local_path"]).exists():
        return data
    return None


def _write_checkpoint(cfg: AppConfig, res: ForgeResult) -> None:
    p = _checkpoint_path(cfg, res.device_id, res.day, res.source_idx)
    atomic_write_text(p, json.dumps({
        "device_id": res.device_id, "source_idx": res.source_idx, "day": res.day,
        "local_path": res.local_path, "hook": res.hook.hook,
        "hook_score": res.hook.score, "hook_attempts": res.hook.attempts,
        "hashtags": res.script.get("hashtags", []), "script": res.script,
        "realism_score": res.realism_score,
        "scenes": [s.to_dict() for s in res.scenes],
        "status": res.status, "error": res.error, "ts": time.time(),
    }, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Forge
# ---------------------------------------------------------------------------

class Forge:
    def __init__(self, cfg: AppConfig, registry: DevicesRegistry | None = None):
        self.cfg = cfg
        self.registry = registry or DevicesRegistry.from_config(cfg)
        self.checker = RealismChecker(min_score=cfg.forge.min_realism_score)
        self.together_key = os.getenv("TOGETHER_API_KEY") or None
        if not self.together_key:
            log.info("TOGETHER_API_KEY не задан — сценарии из запасных шаблонов, хуки из плана mimic")
        self._topics_cache: dict[str, list[str]] = {}
        self._expanded_cache: dict[tuple[str, int], list[str]] = {}
        self._cohorts: dict[str, list[str]] | None = None
        self._lock = threading.Lock()

    # --- когорта и темы ----------------------------------------------------

    def _cohort(self, niche: str) -> list[str]:
        with self._lock:
            if self._cohorts is None:
                try:
                    self._cohorts = niche_cohorts(self.registry.get_active_devices(device_ids=None))
                except Exception as e:
                    log.warning("forge: когорты не построены (%s) — темы без разбиения", e)
                    self._cohorts = {}
            return list(self._cohorts.get(niche, []))

    def _niche_topics(self, niche: str) -> list[str]:
        with self._lock:
            if niche in self._topics_cache:
                return self._topics_cache[niche]
        kws = _niche_keywords(self.cfg, niche)
        topics: list[str] = []
        sources = []
        if self.cfg.forge.use_trend_analyzer:
            sources.append(_load_trend_analyzer_trends(self.cfg, niche))
        sources.append(_load_trends_cache(self.cfg, niche))
        for items in sources:
            for item in items:
                d = _clean_text(item.get("desc") or item.get("text") or "")
                if len(d) >= 8 and _niche_matches(d, kws):
                    topics.append(d)
            if topics:
                break
        topics = list(dict.fromkeys(topics))[:20]
        with self._lock:
            self._topics_cache[niche] = topics
        return topics

    def _expanded_topics(self, niche: str, base: list[str], need: int) -> list[str]:
        key = (niche, need)
        with self._lock:  # один LLM-вызов на нишу, даже при параллельных воркерах
            if key in self._expanded_cache:
                return self._expanded_cache[key]
            seeds = base[:10] or [f"{a} in a {s}" for a, s in FALLBACK_ACTIONS.get(niche, [])] \
                or [niche.replace("_female", "")]
            pool = list(base)
            if self.together_key and len(pool) < need:
                pool += _llm_topic_variations(niche, seeds, need - len(pool), self.together_key,
                                              self.cfg.forge.llm_model)
            # детерминированная добивка: тема × угол подачи
            for angle in TOPIC_ANGLES:
                for seed in seeds:
                    if len(pool) >= need:
                        break
                    pool.append(f"{seed} — {angle}")
            pool = list(dict.fromkeys(pool))
            if len(pool) < need:
                log.info("forge %s: тем %d < %d — у части аккаунтов порция меньше %d",
                         niche, len(pool), need, TOPICS_PER_DEVICE)
            self._expanded_cache[key] = pool
            return pool

    def _topics_for(self, niche: str, device_id: str | None = None) -> list[str]:
        """Без device_id — общие темы ниши (как раньше). С device_id — своя порция:
        pool[rank::N], порции разных аккаунтов ниши не пересекаются."""
        base = self._niche_topics(niche)
        if device_id is None:
            return base
        rank, n = cohort_rank(self._cohort(niche), device_id)
        need = n * TOPICS_PER_DEVICE
        pool = base if len(base) >= need else self._expanded_topics(niche, base, need)
        if not pool:
            return []
        return pool[rank::n] or [pool[rank % len(pool)]]

    # --- сценарий --------------------------------------------------------------

    def _generate_script_with_retry(self, device_id: str, niche: str, topics: list[str],
                                    day: str, source_idx: int,
                                    persona_angle: str = "") -> tuple[dict, HookResult]:
        rnd = random.Random(_stable_int(device_id, day, source_idx, "script"))
        best_script: dict | None = None
        best_hook = HookResult(hook="", score=-1, attempts=0, ok=False)
        # без LLM повторять нечего: запасные хуки детерминированы
        max_attempts = self.cfg.forge.hook_max_retries + 1 if self.together_key else 1
        threshold = self.cfg.forge.hook_score_threshold
        base = _stable_int(device_id, day, source_idx)
        for i in range(max_attempts):
            # каждое видео и каждая попытка получают свою тему, а не всегда topics[0]
            topic = topics[(base + i) % len(topics)] if topics else niche.replace("_female", "")
            script = _llm_script(niche, topic, self.together_key, self.cfg.forge.llm_model,
                                 self.cfg.forge.language, rnd, persona_angle=persona_angle)
            hook = script.get("hook", "")
            sc = score_hook(hook)
            if sc > best_hook.score:
                best_script = script
                best_hook = HookResult(hook=hook, score=sc, attempts=i + 1, ok=sc >= threshold)
            if best_hook.ok:
                break
            log.info("хук '%s' score=%d < %d, попытка %d/%d", hook, sc, threshold, i + 1, max_attempts)
        if best_script is None:
            best_script = _fallback_script(niche, topics, rnd)
            best_hook = HookResult(best_script["hook"], score_hook(best_script["hook"]), max_attempts,
                                   score_hook(best_script["hook"]) >= threshold)
        best_hook.attempts = max(best_hook.attempts, 1)
        return best_script, best_hook

    def _apply_plan_hook(self, device_id: str, day: str, source_idx: int, script: dict,
                         hook: HookResult) -> tuple[dict, HookResult]:
        """Запасной сценарий имеет 2 хука на нишу — у 30 аккаунтов они совпадут.
        В этом случае берём хук из плана mimic: он уникален внутри ниши на день."""
        if script.get("source") != "fallback":
            return script, hook
        try:
            from mimic import get_slot
            plan_hook = str(get_slot(self.cfg, device_id, day, source_idx).get("hook") or "")
        except Exception as e:
            log.warning("план mimic для %s/%s не прочитан: %s", device_id, day, e)
            plan_hook = ""
        if not plan_hook:
            return script, hook
        script = {**script, "hook": plan_hook, "hook_source": "mimic_plan"}
        sc = score_hook(plan_hook)
        return script, HookResult(plan_hook, sc, hook.attempts, sc >= self.cfg.forge.hook_score_threshold)

    def plan_script(self, device: dict, source_idx: int, day: str) -> tuple[dict, HookResult, list[Scene]]:
        """Сценарий + сцены без генерации (используется generate_one и --dry-run)."""
        device_id = validate_device_id(device["device_id"])
        niche = device["niche"]
        topics = self._topics_for(niche, device_id)
        script, hook = self._generate_script_with_retry(device_id, niche, topics, day, source_idx,
                                                        persona_angle=persona_angle(self.cfg, device))
        script, hook = self._apply_plan_hook(device_id, day, source_idx, script, hook)
        rnd = random.Random(_stable_int(device_id, day, source_idx, "scenes"))
        return script, hook, _build_scenes(self.cfg, device, script, rnd)

    def generate_one(self, device: dict, source_idx: int, day: str | None = None) -> ForgeResult:
        from run_generation_gcp import generate_for_device  # импорт здесь: dry-run работает без GCP

        device_id = validate_device_id(device["device_id"])
        day = validate_day(day or self.cfg.today())

        done = _is_done(self.cfg, device_id, day, source_idx)
        if done:
            log.info("forge: %s/%s/source_%d уже готов, пропуск", device_id, day, source_idx)
            return ForgeResult(device_id=device_id, source_idx=source_idx, day=day,
                               local_path=done["local_path"],
                               hook=HookResult(done.get("hook", ""), done.get("hook_score", 0),
                                               done.get("hook_attempts", 0), True),
                               script=done.get("script", {}), status="skipped")

        script, hook, scenes = self.plan_script(device, source_idx, day)
        ok, realism = self.checker.check([s.to_dict() for s in scenes])
        if not ok:
            log.warning("forge %s: realism=%d < %d, отбрасываю", device_id, realism,
                        self.cfg.forge.min_realism_score)
            res = ForgeResult(device_id, source_idx, day, None, hook, script, scenes,
                              realism, "realism_rejected")
            _write_checkpoint(self.cfg, res)
            return res

        voice_id = device.get("voice_id") or _stable_voice_id(
            device_id, self.cfg.uniqueness.get("voice_pool", []))
        split_screen = bool(self.cfg.uniqueness.get("split_screen_enabled", False))
        out_path = self.cfg.videos_root / device_id / day / f"source_{source_idx}.mp4"

        try:
            # persona_lock едет внутри master-сцены: run_generation_gcp кладёт scenes в payload
            # как есть [ПРОВЕРИТЬ по run_generation_gcp.py, что лишние ключи сцены не отбрасываются]
            out = generate_for_device(
                device_id=device_id, scenes=[s.to_dict() for s in scenes],
                voice_id=voice_id, split_screen=split_screen,
                project_id=self.cfg.gcp_project_id,
                videos_root=self.cfg.videos_root, source_idx=source_idx,
                timeout_sec=self.cfg.forge.job_timeout_sec,
                day=day, narration=hook.hook,
                seed=_stable_int(device_id, day, source_idx, "seed"),
                negative_prompt=NEGATIVE_PROMPT, out_path=out_path,
            )
        except Exception as e:
            log.error("generate_for_device %s: %s", device_id, e)
            res = ForgeResult(device_id, source_idx, day, None, hook, script, scenes,
                              realism, "gcp_failed", str(e))
            _write_checkpoint(self.cfg, res)
            return res

        res = ForgeResult(device_id, source_idx, day, out.get("local_path"), hook, script, scenes,
                          realism, out.get("status", "unknown"),
                          None if out.get("status") == "ok" else json.dumps(out, default=str)[:500],
                          # gcp_client.run_job отдаёт имя в ключе "execution"
                          # [ПРОВЕРИТЬ: пробрасывает ли его run_generation_gcp.generate_for_device]
                          gcp_execution_name=out.get("execution") or out.get("execution_name"),
                          # gcp_client отдаёт duration_sec [ПРОВЕРИТЬ: пробрасывает ли run_generation_gcp]
                          gpu_seconds=_float_or_none(out.get("duration_sec") or out.get("gpu_seconds")))
        _write_checkpoint(self.cfg, res)
        return res

    def run(self, device_ids: list[str] | None = None, parallel: bool = False,
            day: str | None = None) -> list[ForgeResult]:
        devices = self.registry.get_active_devices(device_ids=device_ids)
        if not devices:
            log.warning("forge: нет активных устройств")
            return []
        day = day or self.cfg.today()
        tasks = [(d, idx) for d in devices
                 for idx in range(1, self.cfg.forge.videos_per_device_per_day + 1)]

        def _one(device: dict, idx: int) -> ForgeResult:
            try:
                return self.generate_one(device, idx, day)
            except Exception as e:
                log.exception("forge %s/source_%d упал: %s", device["device_id"], idx, e)
                return ForgeResult(device["device_id"], idx, day, None,
                                   HookResult("", 0, 0, False), status="crash", error=str(e))

        results: list[ForgeResult] = []
        workers = max(1, int(self.cfg.forge.max_parallel_jobs)) if parallel else 1
        if workers == 1:
            results = [_one(d, i) for d, i in tasks]
        else:
            # задачи упираются в ожидание Cloud Run Job, поэтому потоков достаточно
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_one, d, i) for d, i in tasks]
                for f in as_completed(futs):
                    results.append(f.result())
            results.sort(key=lambda r: (r.device_id, r.source_idx))
        ok = sum(1 for r in results if r.status in ("ok", "skipped"))
        log.info("forge готово: %d/%d ok", ok, len(results))
        return results


def main() -> None:
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--parallel", action="store_true")
        ap.add_argument("--dry-run", action="store_true",
                        help="показать сценарии и сцены без запуска GCP")

    cfg, a = cli_config("forge", extra=extra)
    forge = Forge(cfg)
    if a.dry_run:
        day = cfg.today()
        out = []
        for d in forge.registry.get_active_devices(device_ids=a.device):
            for idx in range(1, cfg.forge.videos_per_device_per_day + 1):
                script, hook, scenes = forge.plan_script(d, idx, day)
                out.append({"device": d["device_id"], "source_idx": idx,
                            "hook": asdict(hook), "script": script,
                            "scenes": [s.to_dict() for s in scenes]})
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        return
    results = forge.run(a.device, parallel=a.parallel)
    sys.stdout.write(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2,
                                default=str) + "\n")


if __name__ == "__main__":
    main()
