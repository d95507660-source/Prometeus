"""tests/test_account_health.py — допуск к публикации, shadowban и бан на моках API.

Сеть не используется: HTTP подменён FakeHttp, время — FakeClock.
Запуск:  python -m pytest -q tests/test_account_health.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import account_health as ah  # noqa: E402
import autoposter  # noqa: E402
import config_loader  # noqa: E402
import forge_stage  # noqa: E402
import mimic  # noqa: E402

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
TT_TOKEN = "tt-secret-token-123"
IG_TOKEN = "ig-secret-token-456"
YT_KEY = "yt-secret-key-789"
CHANNEL = "UC" + "a" * 22
IG_USER = "17841400000000000"


# ---------------------------------------------------------------------------
# Моки
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: datetime = T0):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw) -> None:
        self.t += timedelta(**kw)


class FakeCfg:
    def __init__(self, root: Path, **health):
        self.root = root
        self.timezone = "UTC"
        self.account_health = config_loader.AccountHealthConfig(enabled=True, **health)
        self.forge = config_loader.ForgeConfig(videos_per_device_per_day=2)
        self.mimic = {"plan_days": 2, "posts_per_account_per_day": 2}
        self.niches = {}
        self.uniqueness = {"platforms": ["tiktok", "instagram", "youtube"]}
        self.raw = {"autoposter": {"max_posts_per_platform_per_run": 5}, "monetization": {}}
        self.clock: FakeClock | None = None

    def path(self, key: str) -> Path:
        return self.root / {"accounts": "data/accounts", "logs": "logs", "videos": "data/videos",
                            "post_plans": "data/post_plans",
                            "forge_checkpoints": "data/forge_checkpoints",
                            "hashtag_seeds": "data/hashtag_seeds.json"}[key]

    @property
    def videos_root(self) -> Path:
        return self.path("videos")

    @property
    def queue_db_path(self) -> Path:
        return self.root / "data" / "queue.db"

    def today(self) -> str:
        return (self.clock() if self.clock else T0).date().isoformat()


@dataclass
class FakeHttp:
    """Ответы по (метод, конец URL). Значение — (код, тело) или функция(params) → (код, тело)."""
    routes: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def __call__(self, method, url, *, headers=None, params=None, json_body=None, timeout=20.0):
        self.calls.append((method, url, dict(params or {}), dict(headers or {})))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.endswith(suffix):
                return resp(params or {}) if callable(resp) else resp
        return 404, {"error": {"message": f"no route {url}"}}


def device(device_id="aria_fight", **acc_extra) -> dict:
    accounts = {
        "tiktok": {"username": "aria.fight", "enabled": True},
        "instagram": {"username": "aria.fight", "enabled": True, "ig_user_id": IG_USER},
        "youtube": {"username": "aria.fight", "enabled": True, "channel_id": CHANNEL},
    }
    for p, extra in acc_extra.items():
        accounts[p].update(extra)
    return {"device_id": device_id, "niche": "wrestling_female", "accounts": accounts,
            "adb": {"host": "127.0.0.1", "port": 5555}}


def tiktok_videos(*videos):
    """videos: (возраст в часах от «сейчас», просмотры)."""
    def resp(params):
        return 200, {"data": {"videos": [
            {"id": f"v{i}", "create_time": int((CLOCK.t - timedelta(hours=h)).timestamp()),
             "view_count": views} for i, (h, views) in enumerate(videos)]},
            "error": {"code": "ok", "message": ""}}
    return resp


CLOCK = FakeClock()


@pytest.fixture
def env(tmp_path, monkeypatch):
    global CLOCK
    CLOCK = FakeClock()
    cfg = FakeCfg(tmp_path)
    cfg.clock = CLOCK
    http = FakeHttp()
    monkeypatch.setenv("TIKTOK_TOKEN_ARIA_FIGHT", TT_TOKEN)
    monkeypatch.setenv("INSTAGRAM_TOKEN_ARIA_FIGHT", IG_TOKEN)
    monkeypatch.setenv("YOUTUBE_API_KEY", YT_KEY)
    health = ah.AccountHealth(cfg, http=http, now=CLOCK)
    yield cfg, http, health
    for h in list(ah.log.handlers):  # каждый тест — свой tmp-лог
        ah.log.removeHandler(h)
        h.close()


# ---------------------------------------------------------------------------
# Статус new → active и прогрессия постов
# ---------------------------------------------------------------------------

def test_new_account_blocked_then_active_with_progression(env):
    cfg, _, health = env
    d = device()
    seen = []
    for day in range(11):
        ok, _ = health.can_post("aria_fight", "tiktok", device=d)
        seen.append((ok, health.posts_allowed("aria_fight", "tiktok", device=d)))
        CLOCK.advance(days=1)
    assert seen[:7] == [(False, 0)] * 7  # дни 0–6: новый аккаунт
    assert [n for _, n in seen[7:]] == [1, 2, 3, 3]  # день 7 — активация, дальше 1, 2, 3, 3
    st = json.loads(health.state_path("aria_fight", "tiktok").read_text(encoding="utf-8"))
    assert st["status"] == "active" and st["created_source"] == "first_seen"


def test_created_at_from_devices_json(env):
    _, _, health = env
    d = device(tiktok={"created_at": "2026-01-01"})
    assert health.can_post("aria_fight", "tiktok", device=d) == (True, "active")
    assert health.posts_allowed("aria_fight", "tiktok", device=d) == 3
    assert health.can_post("aria_fight", "instagram", device=d)[0] is False  # у IG даты нет


def test_existing_account_inferred_from_api_posts(env):
    _, http, health = env
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((24 * 40, 900), (24 * 30, 700))
    st = health.check_account(device(), "tiktok")
    assert st["created_source"] == "api_earliest_post" and st["status"] == "active"
    assert st["first_post_at"] is not None


# ---------------------------------------------------------------------------
# Shadowban
# ---------------------------------------------------------------------------

def test_tiktok_low_views_after_48h_sets_suspect_and_pauses(env):
    cfg, http, health = env
    d = device(tiktok={"created_at": "2026-01-01"})
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((50, 10), (60, 30), (5, 0))
    st = health.check_account(d, "tiktok")
    assert st["shadowban_suspect"] is True and st["avg_views_48h"] == 20.0
    assert st["status"] == "restricted"
    assert health.can_post("aria_fight", "tiktok", device=d)[0] is False
    assert health.posts_allowed("aria_fight", "tiktok", device=d) == 0
    assert "tiktok" not in health.plan_platforms(d)
    # охват вырос → пауза снимается
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((50, 120), (60, 80))
    st = health.check_account(d, "tiktok")
    assert st["shadowban_suspect"] is False and st["status"] == "active"


def test_fresh_posts_younger_than_48h_are_not_judged(env):
    _, http, health = env
    d = device(tiktok={"created_at": "2026-01-01"})
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((3, 0), (20, 1), (47, 2))
    st = health.check_account(d, "tiktok")
    assert st["shadowban_suspect"] is False and st["avg_views_48h"] is None


def test_instagram_insights_both_response_shapes(env):
    _, http, health = env
    d = device(instagram={"created_at": "2026-01-01"})
    ts = lambda h: (CLOCK.t - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S+0000")  # noqa: E731
    http.routes[("GET", f"/{IG_USER}/media")] = (200, {"data": [
        {"id": "111", "timestamp": ts(50)}, {"id": "222", "timestamp": ts(70)},
        {"id": "333", "timestamp": ts(2)}]})
    http.routes[("GET", "/111/insights")] = (200, {"data": [{"name": "views", "values": [{"value": 20}]}]})
    http.routes[("GET", "/222/insights")] = (200, {"data": [{"name": "views", "total_value": {"value": 40}}]})
    st = health.check_account(d, "instagram")
    assert st["avg_views_48h"] == 30.0 and st["shadowban_suspect"] is True
    insight_calls = [c for c in http.calls if c[1].endswith("/insights")]
    assert len(insight_calls) == 2  # для свежего поста 333 insights не запрашивались
    assert all("/v21.0/" in c[1] for c in http.calls)


def test_youtube_chain_and_low_views(env):
    _, http, health = env
    d = device(youtube={"created_at": "2026-01-01"})
    iso = lambda h: (CLOCK.t - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    http.routes[("GET", "/youtube/v3/channels")] = (200, {"items": [
        {"contentDetails": {"relatedPlaylists": {"uploads": "UUaaaa"}}}]})
    http.routes[("GET", "/youtube/v3/playlistItems")] = (200, {"items": [
        {"contentDetails": {"videoId": "vidA", "videoPublishedAt": iso(49)}},
        {"contentDetails": {"videoId": "vidB", "videoPublishedAt": iso(10)}}]})
    http.routes[("GET", "/youtube/v3/videos")] = lambda p: (200, {"items": [
        {"id": vid, "statistics": {"viewCount": "12"}} for vid in p["id"].split(",")]})
    st = health.check_account(d, "youtube")
    videos_call = [c for c in http.calls if c[1].endswith("/videos")][0]
    assert videos_call[2]["id"] == "vidA"
    assert st["shadowban_suspect"] is True and st["avg_views_48h"] == 12.0


# ---------------------------------------------------------------------------
# Бан и ошибки
# ---------------------------------------------------------------------------

def test_suspended_error_bans_and_is_sticky(env):
    _, http, health = env
    d = device(tiktok={"created_at": "2026-01-01"})
    http.routes[("POST", "/v2/video/list/")] = (403, {"error": {
        "code": "access_denied", "message": "The account is suspended"}})
    st = health.check_account(d, "tiktok")
    assert st["status"] == "banned" and "suspended" in st["status_reason"]
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((50, 999))
    assert health.check_account(d, "tiktok")["status"] == "banned"  # без ручного снятия не меняется
    assert not any(c[1].endswith("/v2/video/list/") for c in http.calls[1:])  # API больше не дёргаем
    assert health.can_post("aria_fight", "tiktok", device=d)[0] is False
    assert health.set_status("aria_fight", "tiktok", "active")["status"] == "active"


def test_auth_error_does_not_change_status(env):
    _, http, health = env
    d = device(tiktok={"created_at": "2026-01-01"})
    http.routes[("POST", "/v2/video/list/")] = (401, {"error": {
        "code": "access_token_invalid", "message": "The access token is invalid"}})
    st = health.check_account(d, "tiktok")
    assert st["status"] == "active" and st["last_error"].startswith("auth")


def test_network_error_does_not_change_status(env):
    cfg, _, _ = env

    def boom(*a, **kw):
        raise ConnectionError("network down")

    health = ah.AccountHealth(cfg, http=boom, now=CLOCK)
    st = health.check_account(device(tiktok={"created_at": "2026-01-01"}), "tiktok")
    assert st["status"] == "active" and "ConnectionError" in st["last_error"]


def test_youtube_channel_missing_bans_only_if_seen_before(env):
    _, http, health = env
    d = device(youtube={"created_at": "2026-01-01"})
    http.routes[("GET", "/youtube/v3/channels")] = (200, {"items": []})
    assert health.check_account(d, "youtube")["status"] == "active"  # мог быть неверный id
    http.routes[("GET", "/youtube/v3/channels")] = (200, {"items": [
        {"contentDetails": {"relatedPlaylists": {"uploads": "UUaaaa"}}}]})
    http.routes[("GET", "/youtube/v3/playlistItems")] = (200, {"items": []})
    assert health.check_account(d, "youtube")["verified"] is True
    http.routes[("GET", "/youtube/v3/channels")] = (200, {"items": []})
    assert health.check_account(d, "youtube")["status"] == "banned"


def test_missing_credentials_and_bad_ids_make_no_requests(env, monkeypatch):
    _, http, health = env
    monkeypatch.delenv("TIKTOK_TOKEN_ARIA_FIGHT")
    d = device(instagram={"ig_user_id": "../../me"}, youtube={"channel_id": "x?key=1"})
    for p in ("tiktok", "instagram", "youtube"):
        st = health.check_account(d, p)
        assert st["last_error"].startswith("not_configured")
    assert http.calls == []
    assert "aria_fight/tiktok: TIKTOK_TOKEN_ARIA_FIGHT" in health.missing_credentials([d])


def test_log_file_has_device_account_and_no_secrets(env):
    cfg, http, health = env
    http.routes[("POST", "/v2/video/list/")] = tiktok_videos((50, 10))
    http.routes[("GET", "/youtube/v3/channels")] = (500, {"error": {"code": 500, "message": "backend"}})
    health.run([device(tiktok={"created_at": "2026-01-01"})])
    for h in ah.log.handlers:
        h.flush()
    text = (cfg.path("logs") / "account_health.log").read_text(encoding="utf-8")
    assert "device=aria_fight account=tiktok" in text and "shadowban_suspect" in text
    assert "duration=" in text
    for secret in (TT_TOKEN, IG_TOKEN, YT_KEY):
        assert secret not in text


def test_disabled_module_allows_everything(tmp_path):
    cfg = FakeCfg(tmp_path)
    cfg.account_health.enabled = False
    health = ah.AccountHealth(cfg, http=FakeHttp())
    assert health.can_post("aria_fight", "tiktok") == (True, "account_health выключен")
    assert health.posts_allowed("aria_fight", "tiktok") == ah.UNLIMITED
    assert ah.health_for(cfg) is None
    for h in list(ah.log.handlers):
        ah.log.removeHandler(h)
        h.close()


# ---------------------------------------------------------------------------
# Интеграция: mimic, autoposter, forge_stage
# ---------------------------------------------------------------------------

@dataclass
class FakeRegistry:
    devices: list

    def get_active_devices(self, niches=None, device_ids=None):
        return [d for d in self.devices if not device_ids or d["device_id"] in device_ids]


def test_mimic_skips_banned_and_suspect_platforms(env):
    cfg, _, health = env
    d1, d2 = device("aria_fight"), device("mia_fight")
    health.set_status("aria_fight", "tiktok", "banned")
    st = health.load_state("aria_fight", "youtube")
    st["shadowban_suspect"] = True
    health.save_state(st)
    for p in ("tiktok", "instagram", "youtube"):
        health.set_status("mia_fight", p, "banned")
    out = mimic.Mimic(cfg, FakeRegistry([d1, d2]), health=health).run()
    assert out["mia_fight"] == {"skipped": "все аккаунты banned или shadowban_suspect"}
    assert not mimic.plan_file(cfg, "mia_fight").exists()
    plan = mimic.load_plan(cfg, "aria_fight")
    assert plan["platforms"] == ["instagram"]


def _make_videos(cfg, device_id: str, day: str, n_sources: int = 2) -> None:
    for i in range(1, n_sources + 1):
        sd = cfg.videos_root / device_id / day / f"source_{i}"
        sd.mkdir(parents=True)
        for p in ("tiktok", "instagram", "youtube"):
            (sd / f"{p}.mp4").write_bytes(b"x")


def test_autoposter_posts_only_active_within_progression(env):
    cfg, _, health = env
    day = cfg.today()
    d = device(tiktok={"created_at": "2026-11-03"},     # активация сегодня → 1 пост
               instagram={"created_at": "2026-01-01"})  # давно активен → 3 поста
    health.set_status("aria_fight", "youtube", "banned")
    _make_videos(cfg, "aria_fight", day)
    res = autoposter.post_device(d, cfg, day, dry_run=True, health=health)
    by_platform: dict[str, int] = {}
    for item in res["planned"]:
        by_platform[item["platform"]] = by_platform.get(item["platform"], 0) + 1
    assert by_platform == {"tiktok": 1, "instagram": 2}
    assert res["blocked"]["youtube"].startswith("banned")


def test_autoposter_respects_posts_already_made_today(env):
    cfg, _, health = env
    day = cfg.today()
    d = device(tiktok={"created_at": "2026-11-03"})
    _make_videos(cfg, "aria_fight", day)
    (cfg.videos_root / "aria_fight" / day / "source_1" / "posted.json").write_text(
        json.dumps({"tiktok": {"ok": True}}), encoding="utf-8")
    res = autoposter.post_device(d, cfg, day, dry_run=True, health=health)
    assert [p for p in res["planned"] if p["platform"] == "tiktok"] == []


def test_autoposter_new_account_never_connects_adb(env, monkeypatch):
    cfg, _, health = env
    day = cfg.today()
    _make_videos(cfg, "aria_fight", day)

    def no_adb(*a, **kw):
        raise AssertionError("не должен подключаться к телефону")

    monkeypatch.setattr(autoposter, "connect", no_adb)
    res = autoposter.post_device(device(), cfg, day, dry_run=False, health=health)
    assert res["posted"] == 0 and set(res["blocked"]) == {"tiktok", "instagram", "youtube"}


def test_forge_stage_skips_fully_banned_devices(env, monkeypatch):
    cfg, _, health = env
    for p in ("tiktok", "instagram", "youtube"):
        health.set_status("mia_fight", p, "banned")
    monkeypatch.setattr(ah, "health_for", lambda c: health)

    class NoopForge:
        def generate_one(self, dev, idx, day):
            from types import SimpleNamespace
            return SimpleNamespace(status="ok", error=None, gcp_execution_name=None)

    code = forge_stage.stage_forge(cfg, FakeRegistry([device("aria_fight"), device("mia_fight")]),
                                   forge=NoopForge(), settings=forge_stage.RunSettings(
                                       max_workers=2, poll_sec=0.01, heartbeat_sec=0.2))
    from job_queue import JobQueue
    assert code == 0
    assert {j.device_id for j in JobQueue(cfg.queue_db_path).jobs()} == {"aria_fight"}


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    return tmp_path / "config.json"


def test_config_account_health_loaded(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="config_loader"):
        cfg = config_loader.load_config(_write_cfg(tmp_path))
    assert not any("неизвестные ключи" in r.getMessage() for r in caplog.records)
    h = cfg.account_health
    assert (h.enabled, h.new_account_days, h.max_posts_per_day_progression,
            h.shadowban_views_threshold, h.shadowban_check_hours) == (True, 7, [1, 2, 3], 50, 48)
    assert cfg.path("accounts") == tmp_path / "data" / "accounts"


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["account_health"].update(max_posts_per_day_progression=[]), "progression"),
    (lambda r: r["account_health"].update(shadowban_check_hours=200), "shadowban_check_hours"),
    (lambda r: r["account_health"].update(graph_api_version="latest"), "graph_api_version"),
])
def test_config_account_health_validation(tmp_path, mutate, msg):
    with pytest.raises(ValueError, match=msg):
        config_loader.load_config(_write_cfg(tmp_path, mutate))


def test_config_without_block_keeps_old_behaviour(tmp_path):
    cfg = config_loader.load_config(_write_cfg(tmp_path, lambda r: r.pop("account_health")))
    assert cfg.account_health.enabled is False and ah.health_for(cfg) is None
