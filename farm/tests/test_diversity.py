"""tests/test_diversity.py — критерии приёмки разнообразия на моках.

Без GCP, без GPU, без LLM. ffmpeg нужен только для test_ghost_filters_run_in_ffmpeg
(тест пропускается, если ffmpeg не установлен).

Запуск:  python -m pytest -q tests/test_diversity.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ensure_module(name: str, **attrs) -> None:
    """Если настоящего модуля нет (CI без кода пользователя) — подставляем заглушку."""
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod


class _StubRegistry:
    @classmethod
    def from_config(cls, cfg):  # pragma: no cover - в тестах всегда передаём FakeRegistry
        raise RuntimeError("заглушка: передай registry явно")


_ensure_module("config_loader", AppConfig=object, cli_config=None)
_ensure_module("devices_registry", DevicesRegistry=_StubRegistry)

import forge  # noqa: E402
import ghost  # noqa: E402
import mimic  # noqa: E402
import model_server  # noqa: E402

NICHE = "fitness_female"
N_ACCOUNTS = 30
DAYS = 7
POSTS_PER_DAY = 2
START = "2026-11-10"
TAGS = ["#fitness", "#gym", "#workout", "#fitgirl", "#legday", "#homeworkout",
        "#strength", "#fitnessmotivation", "#training", "#healthy", "#cardio", "#abs"]


# ---------------------------------------------------------------------------
# Моки
# ---------------------------------------------------------------------------

@dataclass
class FakeForgeSettings:
    language: str = "en"
    scenes_per_video: tuple[int, int] = (2, 3)
    video_duration_sec: tuple[int, int] = (10, 15)
    scene_duration_sec: int = 5
    min_realism_score: int = 70
    hook_max_retries: int = 2
    hook_score_threshold: int = 50
    llm_model: str = "test-model"
    use_trend_analyzer: bool = True
    videos_per_device_per_day: int = POSTS_PER_DAY
    max_parallel_jobs: int = 1
    job_timeout_sec: int = 60


class FakeCfg:
    def __init__(self, root: Path, language: str = "en"):
        self.root = root
        self.niches = {n: {"hashtags_seed": TAGS} for n in mimic.NICHE_HOOKS}
        self.uniqueness = {"platforms": ["tiktok", "instagram", "youtube"], "voice_pool": []}
        self.mimic = {"plan_days": DAYS, "posts_per_account_per_day": POSTS_PER_DAY}
        self.forge = FakeForgeSettings(language=language)
        self.videos_root = root / "videos"
        self.gcp_project_id = "test-project"

    def path(self, name: str) -> Path:
        return self.root / name

    def today(self) -> str:
        return START


@dataclass
class FakeRegistry:
    devices: list[dict] = field(default_factory=list)

    def get_active_devices(self, device_ids=None):
        if device_ids:
            return [d for d in self.devices if d["device_id"] in device_ids]
        return list(self.devices)


def make_devices(niche: str = NICHE, n: int = N_ACCOUNTS) -> list[dict]:
    accounts = {p: {"enabled": True, "username": f"user_{p}"} for p in ("tiktok", "instagram", "youtube")}
    return [{"device_id": f"dev_{niche[:3]}_{i:02d}", "niche": niche, "accounts": accounts}
            for i in range(1, n + 1)]


@pytest.fixture
def cfg(tmp_path: Path) -> FakeCfg:
    return FakeCfg(tmp_path)


@pytest.fixture
def devices() -> list[dict]:
    return make_devices()


@pytest.fixture
def registry(devices) -> FakeRegistry:
    return FakeRegistry(devices)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)


def _days() -> list[str]:
    d0 = date.fromisoformat(START)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(DAYS)]


def _all_slots(cfg: FakeCfg, devices: list[dict]):
    for dev in devices:
        plan = mimic.load_plan(cfg, dev["device_id"])
        for day in plan["days"]:
            for slot in day["slots"]:
                yield dev["device_id"], day["date"], slot


# ---------------------------------------------------------------------------
# mimic: хуки, хэштеги, CTA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["en", "ru"])
@pytest.mark.parametrize("niche", sorted(mimic.NICHE_HOOKS))
def test_hooks_unique_per_day_within_niche(tmp_path, language, niche):
    cfg = FakeCfg(tmp_path, language=language)
    devs = make_devices(niche)
    mimic.Mimic(cfg, FakeRegistry(devs)).run()
    by_day: dict[str, list[str]] = defaultdict(list)
    for _, day, slot in _all_slots(cfg, devs):
        by_day[day].append(slot["hook"])
    assert sorted(by_day) == _days()
    for day, hooks in by_day.items():
        assert len(hooks) == N_ACCOUNTS * POSTS_PER_DAY
        dupes = {h for h in hooks if hooks.count(h) > 1}
        assert not dupes, f"{niche}/{language} {day}: повторы хуков {dupes}"


def test_hooks_match_language(tmp_path):
    for language, want_cyr in (("en", False), ("ru", True)):
        cfg = FakeCfg(tmp_path / language, language=language)
        pool = mimic.hook_pool(cfg, NICHE)
        assert all(bool(mimic._CYR.search(h)) == want_cyr for h in pool)
        assert all(len(h) <= mimic.HOOK_MAX_LEN for h in pool)
        assert all("{" not in h for h in pool), "непрошедшая подстановка в шаблоне"


def test_hashtag_combinations_never_repeat(cfg, devices, registry):
    mimic.Mimic(cfg, registry).run()
    seen: dict[frozenset, tuple] = {}
    for device_id, day, slot in _all_slots(cfg, devices):
        tags = slot["hashtags"]
        assert 4 <= len(tags) <= 5
        assert set(tags) <= set(TAGS[:mimic.HASHTAG_POOL_SIZE]), "теги вне пула из 10"
        key = frozenset(tags)
        assert key not in seen, f"{device_id} {day}: те же теги, что у {seen[key]}"
        seen[key] = (device_id, day, slot["slot"])
    assert len(seen) == N_ACCOUNTS * DAYS * POSTS_PER_DAY


def test_cta_from_three_and_deterministic(cfg, devices, registry):
    m = mimic.Mimic(cfg, registry)
    m.run()
    first = {(d, day, s["slot"]): (s["hook"], s["cta"], tuple(s["hashtags"]))
             for d, day, s in _all_slots(cfg, devices)}
    m.run(replan=True)
    second = {(d, day, s["slot"]): (s["hook"], s["cta"], tuple(s["hashtags"]))
              for d, day, s in _all_slots(cfg, devices)}
    assert first == second, "план должен быть детерминированным"
    ctas = {v[1] for v in first.values()}
    assert ctas <= set(mimic.NICHE_CTAS[NICHE]) and len(ctas) == 3
    # в один день у одного слота CTA распределены по аккаунтам, а не одинаковы у всех
    day0 = [v[1] for k, v in first.items() if k[1] == START and k[2] == 1]
    assert len(set(day0)) == 3


def test_partial_run_uses_full_cohort(cfg, devices, registry):
    """--device dev_X не должен менять ранг (иначе хуки совпадут с соседями)."""
    m = mimic.Mimic(cfg, registry)
    m.run()
    full = mimic.load_plan(cfg, devices[5]["device_id"])
    m.run(device_ids=[devices[5]["device_id"]], replan=True)
    single = mimic.load_plan(cfg, devices[5]["device_id"])
    assert full["days"] == single["days"]
    assert single["cohort_size"] == N_ACCOUNTS


def test_replan_keeps_published_slots(cfg, devices, registry):
    m = mimic.Mimic(cfg, registry)
    m.run()
    f = mimic.plan_file(cfg, devices[0]["device_id"])
    plan = json.loads(f.read_text(encoding="utf-8"))
    plan["days"][0]["slots"][0].update(status="posted", hook="CUSTOM")
    f.write_text(json.dumps(plan), encoding="utf-8")
    m.run(replan=True)
    assert mimic.get_slot(cfg, devices[0]["device_id"], START, 1)["hook"] == "CUSTOM"


def test_build_plans_backward_compatible_signature(cfg, devices):
    plan = mimic.build_plans(cfg, devices[0], 2, 2)  # без cohort, как в старом коде
    assert len(plan["days"]) == 2 and plan["cohort_size"] == 1


# ---------------------------------------------------------------------------
# Лицо: persona_lock
# ---------------------------------------------------------------------------

def test_persona_lock_unique_and_stable(cfg, devices):
    locks = [forge.persona_lock(cfg, d["device_id"]) for d in devices]
    assert len(set(locks)) == len(locks)
    assert locks == [forge.persona_lock(cfg, d["device_id"]) for d in devices]
    # совместимость: та же формула, что была в model_server → старые лица не меняются
    for d, lock in zip(devices, locks):
        assert lock == int(hashlib.sha256(d["device_id"].encode()).hexdigest()[:8], 16)


def test_persona_lock_from_metadata(cfg, devices):
    did = devices[0]["device_id"]
    meta = cfg.path("models") / did / "metadata.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"persona_lock": 424242}), encoding="utf-8")
    assert forge.persona_lock(cfg, did) == 424242
    meta.write_text(json.dumps({"persona_lock": -5}), encoding="utf-8")
    assert forge.persona_lock(cfg, did) != -5


def test_master_scene_carries_lock_to_model_server(cfg, devices):
    for d in devices:
        script, _, scenes = forge.Forge(cfg, FakeRegistry(devices)).plan_script(d, 1, START)
        payload = {"scenes": [s.to_dict() for s in scenes]}
        assert model_server._persona_seed(payload, d["device_id"]) == forge.persona_lock(cfg, d["device_id"])
        break


def test_model_server_persona_seed_priority():
    did = "dev_x"
    legacy = int(hashlib.sha256(did.encode()).hexdigest()[:8], 16)
    assert model_server._persona_seed({"scenes": []}, did) == legacy
    scenes = [{"role": "master", "prompt": "p", "persona_lock": 7}, {"role": "scene", "prompt": "q"}]
    assert model_server._persona_seed({"scenes": scenes}, did) == 7
    assert model_server._persona_seed({"scenes": scenes, "persona_lock": 9}, did) == 9
    assert model_server._persona_seed({"scenes": scenes, "persona_lock": True}, did) == 7
    assert model_server._persona_seed({"scenes": [], "persona_lock": 2 ** 40}, did) == legacy


def test_model_server_rejects_bad_device_id(monkeypatch):
    monkeypatch.setenv("DEVICE_ID", "../etc")
    monkeypatch.setenv("SCENES_URI", "gs://b/k.json")
    monkeypatch.setenv("OUTPUT_URI", "gs://b/out.mp4")
    with pytest.raises(ValueError):
        model_server.run_job()


# ---------------------------------------------------------------------------
# forge: внешность, сцены, темы, persona_angle
# ---------------------------------------------------------------------------

def test_appearance_distinct_without_metadata(cfg, devices):
    looks = [forge.auto_appearance(d["device_id"]) for d in devices]
    assert len(set(looks)) == len(looks)
    masters = []
    for d in devices:
        scenes = forge._build_scenes(cfg, d, {"action": "a", "setting": "s"},
                                     forge.random.Random(1))
        masters.append(scenes[0].prompt)
        assert "distinct look of" not in scenes[0].prompt
    assert len(set(masters)) == len(masters)


def test_metadata_appearance_wins(cfg, devices):
    did = devices[0]["device_id"]
    meta = cfg.path("models") / did / "metadata.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"appearance": "short silver hair, violet eyes"}), encoding="utf-8")
    scenes = forge._build_scenes(cfg, devices[0], {}, forge.random.Random(1))
    assert "short silver hair, violet eyes" in scenes[0].prompt


def test_scenes_deterministic_per_account_and_differ(cfg, devices, registry):
    f = forge.Forge(cfg, registry)
    a1 = [s.prompt for s in f.plan_script(devices[0], 1, START)[2]]
    a2 = [s.prompt for s in forge.Forge(cfg, registry).plan_script(devices[0], 1, START)[2]]
    assert a1 == a2
    all_video_scenes = [tuple(s.prompt for s in f.plan_script(d, 1, START)[2][1:]) for d in devices]
    assert len(set(all_video_scenes)) == len(all_video_scenes)
    ok, _ = f.checker.check([s.to_dict() for s in f.plan_script(devices[0], 1, START)[2]])
    assert ok, "новые промпты должны проходить RealismChecker"


def test_topics_disjoint_between_accounts(cfg, devices, registry):
    f = forge.Forge(cfg, registry)
    portions = [f._topics_for(NICHE, d["device_id"]) for d in devices]
    assert all(portions)
    flat = [t for p in portions for t in p]
    assert len(flat) == len(set(flat)), "порции тем пересекаются"
    assert f._topics_for(NICHE) == []  # без device_id — старое поведение (общие темы ниши)


def test_topics_from_trend_analysis_are_split(cfg, devices, registry):
    trends = [{"desc": f"gym workout trend number {i}"} for i in range(100)]
    p = cfg.path("trend_analysis") / f"{NICHE}.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(trends), encoding="utf-8")
    f = forge.Forge(cfg, registry)
    portions = [set(f._topics_for(NICHE, d["device_id"])) for d in devices]
    for i, a in enumerate(portions):
        for b in portions[i + 1:]:
            assert not (a & b)


def test_llm_prompt_contains_persona_angle(cfg, devices, registry, monkeypatch):
    prompts: list[str] = []

    def fake_chat(api_key, model, prompt, max_tokens=300, temperature=0.9):
        prompts.append(prompt)
        if '"topics"' in prompt:
            return json.dumps({"topics": [f"idea number {i}" for i in range(200)]})
        return json.dumps({"hook": "Can you do 20 squats with me?", "action": "squatting",
                           "setting": "gym", "hashtags": ["#gym"]})

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setattr(forge, "_together_chat", fake_chat)
    f = forge.Forge(cfg, registry)
    script, hook, _ = f.plan_script(devices[0], 1, START)
    angle = forge.persona_angle(cfg, devices[0])
    assert script["source"] == "llm" and hook.ok
    assert any(angle in p for p in prompts)
    assert sum('"topics"' in p for p in prompts) == 1  # переформулировки — один вызов на нишу


def test_generate_one_end_to_end_uses_unique_plan_hooks(cfg, devices, registry, monkeypatch):
    """Без LLM запасной хук один на нишу → forge берёт уникальный хук из плана mimic."""
    calls: list[dict] = []

    def fake_generate_for_device(**kw):
        calls.append(kw)
        kw["out_path"].parent.mkdir(parents=True, exist_ok=True)
        kw["out_path"].write_bytes(b"fake")
        return {"status": "ok", "local_path": str(kw["out_path"])}

    monkeypatch.setitem(sys.modules, "run_generation_gcp",
                        types.SimpleNamespace(generate_for_device=fake_generate_for_device))
    mimic.Mimic(cfg, registry).run()
    results = forge.Forge(cfg, registry).run()
    assert len(results) == N_ACCOUNTS * POSTS_PER_DAY
    assert all(r.status == "ok" for r in results)
    hooks = [r.hook.hook for r in results]
    assert len(set(hooks)) == len(hooks)
    narr = [c["narration"] for c in calls]
    assert len(set(narr)) == len(narr)
    locks = {c["device_id"]: c["scenes"][0]["persona_lock"] for c in calls}
    assert len(set(locks.values())) == N_ACCOUNTS
    ckpt = forge.read_checkpoint(cfg, devices[0]["device_id"], START, 1)
    assert ckpt["status"] == "ok" and ckpt["script"]["hook_source"] == "mimic_plan"


# ---------------------------------------------------------------------------
# ghost: визуальный стиль
# ---------------------------------------------------------------------------

def test_ghost_styles_differ_within_niche(devices):
    cohort = [d["device_id"] for d in devices]
    styles = [ghost.style_for_device(d, cohort) for d in cohort]
    keys = [s.key() for s in styles]
    assert len(set(keys)) == len(keys), "у двух аккаунтов ниши одинаковый стиль"
    filters = [ghost.build_filter(s) for s in styles]
    assert len(set(filters)) == len(filters)
    # соседи по рангу отличаются базовым пресетом, а не только силой/кадром
    assert all(styles[i].preset != styles[i + 1].preset for i in range(len(styles) - 1))
    # стабильность
    assert keys == [ghost.style_for_device(d, cohort).key() for d in cohort]


def test_ghost_style_override_and_fallback(devices):
    did = devices[0]["device_id"]
    assert ghost.style_for_device(did, style_index=5).index == 5
    assert ghost.style_for_device(did).key() == ghost.style_for_device(did).key()
    with pytest.raises(ValueError):
        ghost.style_for_device("bad id; rm -rf /")


def test_ghost_no_hidden_jitter():
    """Стиль один на аккаунт: фильтр не зависит от площадки и не содержит setpts/atempo."""
    for i in range(ghost.STYLE_COMBOS):
        vf = ghost.build_filter(ghost.style_for_index(i))
        assert "setpts" not in vf and "atempo" not in vf


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
def test_ghost_filters_run_in_ffmpeg(tmp_path, devices):
    src_dir = tmp_path / "videos" / devices[0]["device_id"] / START
    src_dir.mkdir(parents=True)
    src = src_dir / "source_1.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=480x832:rate=24:duration=1", "-pix_fmt", "yuv420p",
                    str(src)], check=True, timeout=120)
    for i in range(len(ghost.PRESET_NAMES)):  # каждый пресет хотя бы раз
        style = ghost.style_for_index(i + 30)  # strong + tight — самые сильные значения
        out = ghost.render_styled(src, tmp_path / f"out_{i}", style)
        assert out.stat().st_size > 0
    res = ghost.uniquify_device(devices[0]["device_id"], tmp_path / "videos",
                                ["tiktok", "instagram", "youtube"], day=START,
                                cohort=[d["device_id"] for d in devices])
    assert len(res) == 3
    sizes = {Path(p).read_bytes() for p in res.values()}
    assert len(sizes) == 1, "площадки одного аккаунта получают один и тот же рендер"
