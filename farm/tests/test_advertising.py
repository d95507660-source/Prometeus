"""tests/test_advertising.py — CPA в плане mimic и обязательная маркировка рекламы в autoposter.

Без телефонов и сети: ADB/uiautomator2 подменены, публикация — dry-run или фейковый post_video.
Запуск:  python -m pytest -q tests/test_advertising.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import advertising  # noqa: E402
import autoposter  # noqa: E402
import config_loader  # noqa: E402
import link_rotator  # noqa: E402
import mimic  # noqa: E402

DAY = "2026-11-10"
TAGS = [f"#tag{i}" for i in range(10)]
ADVERTISER = "ООО «ФитАпп», ИНН 7700000000"
ERID = "2VtzqvXYZ12"
LABEL = f"Реклама. {ADVERTISER}. erid: {ERID}"


class FakeCfg:
    def __init__(self, root: Path, **ad):
        self.root = root
        self.timezone = "UTC"
        self.advertising = config_loader.AdvertisingConfig(enabled=True, **ad)
        self.forge = config_loader.ForgeConfig(videos_per_device_per_day=2)
        self.mimic = {"plan_days": 1, "posts_per_account_per_day": 2}
        self.niches = {"fitness_female": {"hashtags_seed": TAGS}}
        self.uniqueness = {"platforms": ["tiktok", "instagram", "youtube"]}
        self.raw = {"autoposter": {"max_posts_per_platform_per_run": 5},
                    "monetization": {"disclosure": {"enabled": True, "text": "AI-generated character",
                                                    "platforms": ["tiktok", "instagram", "youtube"]}}}

    def path(self, key: str) -> Path:
        return self.root / {"post_plans": "data/post_plans", "forge_checkpoints": "data/forge_checkpoints",
                            "hashtag_seeds": "data/hashtag_seeds.json", "logs": "logs"}[key]

    @property
    def videos_root(self) -> Path:
        return self.root / "data" / "videos"

    @property
    def ad_offers_path(self) -> Path:
        return self.root / "data" / "cpa_offers.json"

    def today(self) -> str:
        return DAY


class Registry:
    def __init__(self, devices):
        self.devices = devices

    def get_active_devices(self, niches=None, device_ids=None):
        return [d for d in self.devices if not device_ids or d["device_id"] in device_ids]


def device(device_id="dev_fit_01", account_type="entertainment") -> dict:
    d = {"device_id": device_id, "niche": "fitness_female", "adb": {"host": "127.0.0.1", "port": 5555},
         "accounts": {p: {"username": f"{device_id}.{p}", "enabled": True}
                      for p in ("tiktok", "instagram", "youtube")}}
    if account_type is not None:
        d["account_type"] = account_type
    return d


def write_offers(cfg: FakeCfg, *, erid: str | None = ERID, keywords: list[str] | None = None) -> None:
    offer = {"id": 7, "keywords": keywords or [t[1:] for t in TAGS], "payout": 3.5,
             "url": "https://fit.example/o?ref=1", "advertiser": ADVERTISER, "name": "FitApp"}
    if erid:
        offer["erid"] = erid
    cfg.ad_offers_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ad_offers_path.write_text(json.dumps([offer], ensure_ascii=False), encoding="utf-8")


def make_plan(cfg: FakeCfg, devices: list[dict]) -> dict:
    return mimic.Mimic(cfg, Registry(devices)).run()


def make_videos(cfg: FakeCfg, device_id: str = "dev_fit_01") -> None:
    for i in (1, 2):
        sd = cfg.videos_root / device_id / DAY / f"source_{i}"
        sd.mkdir(parents=True)
        for p in ("tiktok", "instagram", "youtube"):
            (sd / f"{p}.mp4").write_bytes(b"x")


@pytest.fixture
def cfg(tmp_path) -> FakeCfg:
    c = FakeCfg(tmp_path)
    write_offers(c)
    return c


# ---------------------------------------------------------------------------
# mimic: CPA в плане
# ---------------------------------------------------------------------------

def test_entertainment_plan_gets_ad_with_label_and_links(cfg):
    out = make_plan(cfg, [device()])
    assert out["dev_fit_01"]["ad_slots"] == 2
    slot = mimic.get_slot(cfg, "dev_fit_01", DAY, 1)
    ad = slot["ad"]
    assert ad["label"] == LABEL and ad["offer_id"] == 7 and ad["erid"] == ERID
    for p in ("tiktok", "instagram", "youtube"):
        first, link = slot["description"][p].split("\n")
        assert first == LABEL and link == ad["links"][p]
        q = parse_qs(urlsplit(link).query)
        assert q["ref"] == ["1"] and q["utm_source"] == [p] and q["erid"] == [ERID]
        assert q["utm_campaign"] == [f"account_dev_fit_01_{p}"] and q["utm_content"] == ["offer_7"]


@pytest.mark.parametrize("account_type", ["professional", None])
def test_non_entertainment_device_gets_no_ad(cfg, account_type):
    out = make_plan(cfg, [device(account_type=account_type)])
    assert out["dev_fit_01"]["ad_slots"] == 0
    slot = mimic.get_slot(cfg, "dev_fit_01", DAY, 1)
    assert "ad" not in slot and "description" not in slot


def test_no_matching_offer_no_ad(tmp_path):
    cfg = FakeCfg(tmp_path)
    write_offers(cfg, keywords=["crypto"])
    assert make_plan(cfg, [device()])["dev_fit_01"]["ad_slots"] == 0


def test_advertising_disabled_no_ad(tmp_path):
    cfg = FakeCfg(tmp_path)
    cfg.advertising.enabled = False
    write_offers(cfg)
    assert make_plan(cfg, [device()])["dev_fit_01"]["ad_slots"] == 0


def test_dash_in_device_id_is_sanitized_for_links(cfg):
    make_plan(cfg, [device("dev-fit-02")])
    link = mimic.get_slot(cfg, "dev-fit-02", DAY, 1)["ad"]["links"]["tiktok"]
    assert "utm_campaign=account_dev_fit_02_tiktok" in link


# ---------------------------------------------------------------------------
# autoposter: маркировка
# ---------------------------------------------------------------------------

def dry(cfg, dev=None):
    return autoposter.post_device(dev or device(), cfg, DAY, dry_run=True, health=None)


def by_platform(res) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for item in res["planned"]:
        out.setdefault(item["platform"], []).append(item)
    return out


def test_ad_post_has_label_link_and_platform_markers(cfg):
    make_plan(cfg, [device()])
    make_videos(cfg)
    posts = by_platform(dry(cfg))
    tiktok = posts["tiktok"][0]
    assert tiktok["caption"].startswith(LABEL + "\n\n")
    assert "https://fit.example/o?ref=1" in tiktok["caption"] and f"erid={ERID}" in tiktok["caption"]
    assert "#ad #sponsored" in tiktok["caption"] and tiktok["caption"].endswith("AI-generated character")
    assert tiktok["ad"] == {"offer_id": 7, "erid": ERID, "label": LABEL}
    assert tiktok["ad_toggles"] == []  # у TikTok маркеры — только хэштеги
    insta = posts["instagram"][0]
    assert insta["caption"].startswith(LABEL) and insta["ad_toggles"] == [["Paid partnership"]]
    yt = posts["youtube"][0]
    assert yt["caption"].startswith(LABEL + " | ") and len(yt["caption"]) <= 100
    assert yt["ad_toggles"] == [["Includes paid promotion"]]


def test_post_without_cpa_has_no_ad_marking(cfg):
    make_plan(cfg, [device(account_type="professional")])
    make_videos(cfg)
    res = dry(cfg, device(account_type="professional"))
    assert len(res["planned"]) == 6
    for item in res["planned"]:
        cap = item["caption"]
        assert "Реклама" not in cap and "#ad" not in cap and "#sponsored" not in cap and "erid" not in cap
        assert item["ad"] is None and item["ad_toggles"] == []


def test_require_erid_without_erid_blocks_publication(tmp_path, monkeypatch):
    cfg = FakeCfg(tmp_path, require_erid=True)
    write_offers(cfg, erid=None)
    make_plan(cfg, [device()])
    assert mimic.get_slot(cfg, "dev_fit_01", DAY, 1)["ad"]["erid"] is None
    make_videos(cfg)
    res = dry(cfg)
    assert res["planned"] == [] and len(res["ad_errors"]) == 6
    assert all("erid" in e["error"] for e in res["ad_errors"])

    calls = []
    monkeypatch.setattr(autoposter, "connect", lambda host, port: "serial-1")
    monkeypatch.setattr(autoposter, "cleanup_remote", lambda serial: None)
    monkeypatch.setattr(autoposter, "post_video", lambda *a, **kw: calls.append(a) or True)
    monkeypatch.setattr(autoposter, "human_delay", lambda *a, **kw: None)
    res = autoposter.post_device(device(), cfg, DAY, dry_run=False, health=None)
    assert calls == [] and res["posted"] == 0
    posted = json.loads((cfg.videos_root / "dev_fit_01" / DAY / "source_1" / "posted.json").read_text("utf-8"))
    assert posted["tiktok"]["ok"] is False and "erid" in posted["tiktok"]["error"]


def test_require_erid_with_erid_publishes(tmp_path):
    cfg = FakeCfg(tmp_path, require_erid=True)
    write_offers(cfg)
    make_plan(cfg, [device()])
    make_videos(cfg)
    res = dry(cfg)
    assert len(res["planned"]) == 6 and "ad_errors" not in res


def test_cpa_link_in_description_without_ad_block_is_still_labeled(cfg):
    plan_dir = cfg.path("post_plans")
    plan_dir.mkdir(parents=True)
    link = "https://p.example/?utm_source=tiktok&utm_content=offer_9"
    (plan_dir / "dev_fit_01.json").write_text(json.dumps({"days": [{"date": DAY, "slots": [
        {"source_idx": 1, "hook": "Hi", "hashtags": [], "description": f"смотри {link}"}]}]}), encoding="utf-8")
    sd = cfg.videos_root / "dev_fit_01" / DAY / "source_1"
    sd.mkdir(parents=True)
    (sd / "tiktok.mp4").write_bytes(b"x")
    item = dry(cfg)["planned"][0]
    assert item["caption"].startswith("Реклама. рекламодатель не указан.") and link in item["caption"]
    assert "#ad #sponsored" in item["caption"]
    cfg.advertising.require_erid = True
    assert "erid" in dry(cfg)["ad_errors"][0]["error"]


def test_real_publish_passes_toggle_step_and_records_ad(cfg, monkeypatch):
    make_plan(cfg, [device()])
    make_videos(cfg)
    seen = {}

    def fake_post(serial, platform, video, caption, remote, flows, screenshots):
        seen[(remote, platform)] = (caption, flows[platform])
        return True

    monkeypatch.setattr(autoposter, "connect", lambda host, port: "serial-1")
    monkeypatch.setattr(autoposter, "cleanup_remote", lambda serial: None)
    monkeypatch.setattr(autoposter, "post_video", fake_post)
    monkeypatch.setattr(autoposter, "human_delay", lambda *a, **kw: None)
    res = autoposter.post_device(device(), cfg, DAY, dry_run=False, health=None)
    assert res["posted"] == 6
    insta = next(v for (r, p), v in seen.items() if p == "instagram")
    steps = insta[1]
    toggle = next(i for i, s in enumerate(steps) if s.get("ad_toggle"))
    post = next(i for i, s in enumerate(steps) if "post_any" in s)
    assert toggle == post - 1 and steps[toggle]["optional"] is False
    assert steps[toggle]["click_any"] == ["Paid partnership"]
    posted = json.loads((cfg.videos_root / "dev_fit_01" / DAY / "source_1" / "posted.json").read_text("utf-8"))
    assert posted["instagram"]["ad"] == {"offer_id": 7, "erid": ERID, "label": LABEL}


# ---------------------------------------------------------------------------
# Поток публикации: переключатель рекламной метки
# ---------------------------------------------------------------------------

class FakeSel:
    def __init__(self, dev, text):
        self.dev, self.text = dev, text

    @property
    def exists(self):
        return self.text in self.dev.visible

    def click(self):
        self.dev.clicked.append(self.text)
        if self.text in self.dev.post_texts:
            self.dev.visible.discard(self.text)

    def wait(self, timeout=0):
        return True

    def set_text(self, text):
        self.dev.caption = text


class FakeU2:
    def __init__(self, visible):
        self.visible = set(visible)
        self.post_texts = {"Share"}
        self.clicked: list[str] = []
        self.caption = None

    def __call__(self, text=None, description=None, className=None):
        return FakeSel(self, "<edit>" if className else (text or description))

    def press(self, key):
        pass


@pytest.fixture
def no_delay(monkeypatch):
    monkeypatch.setattr(autoposter, "human_delay", lambda *a, **kw: None)
    monkeypatch.setattr(autoposter.time, "sleep", lambda s: None)


def test_flow_clicks_toggle_before_publishing(no_delay):
    steps = autoposter.ad_flow([{"set_caption": True}, {"post_any": ["Share"], "timeout": 1}],
                               ["Paid partnership"], required=True, timeout=0.01)
    d = FakeU2({"Paid partnership", "Share", "<edit>"})
    autoposter.run_flow(d, steps, "Реклама. X.")
    assert d.clicked[-2:] == ["Paid partnership", "Share"] and d.caption == "Реклама. X."


def test_flow_without_toggle_is_not_published(no_delay):
    steps = autoposter.ad_flow([{"set_caption": True}, {"post_any": ["Share"], "timeout": 1}],
                               ["Paid partnership"], required=True, timeout=0.01)
    d = FakeU2({"Share", "<edit>"})
    with pytest.raises(autoposter.PostError, match="Paid partnership"):
        autoposter.run_flow(d, steps, "Реклама. X.")
    assert "Share" not in d.clicked


def test_flow_toggle_optional_when_not_required(no_delay):
    steps = autoposter.ad_flow([{"post_any": ["Share"], "timeout": 1}], ["Paid partnership"],
                               required=False, timeout=0.01)
    d = FakeU2({"Share"})
    autoposter.run_flow(d, steps, "")
    assert d.clicked == ["Share"]


# ---------------------------------------------------------------------------
# Подпись: обязательные части не обрезаются
# ---------------------------------------------------------------------------

DISC = {"monetization": {"disclosure": {"enabled": True, "text": "AI-generated character",
                                        "platforms": ["tiktok", "youtube"]}}}


def test_long_caption_keeps_mandatory_parts():
    link = "https://fit.example/o?utm_content=offer_7&erid=" + ERID
    cap = autoposter.build_caption_text("Хук " * 1000, [f"#t{i}" for i in range(5)], DISC, "tiktok",
                                        cta="Подпишись", ad_label=LABEL, ad_link=link,
                                        ad_tags=["#ad", "#sponsored"])
    assert len(cap) <= 2200 and cap.startswith(LABEL) and link in cap
    assert "#ad #sponsored" in cap and cap.endswith("AI-generated character")


def test_ad_tags_not_duplicated_with_regular_tags():
    cap = autoposter.build_caption_text("hi", ["#ad", "#fit"], DISC, "tiktok", ad_label=LABEL,
                                        ad_tags=["#ad", "#sponsored"])
    assert cap.count("#ad") == 1 and "#fit #ad #sponsored" in cap


def test_youtube_label_too_long_is_refused():
    with pytest.raises(autoposter.PostError, match="YouTube"):
        autoposter.build_caption_text("hi", [], DISC, "youtube", ad_label="Реклама. " + "Х" * 120)


def test_no_ad_caption_unchanged():
    assert autoposter.build_caption_text("Hi", ["#fit"], DISC, "tiktok", cta="Go") == \
        "Hi\n\nGo\n\n#fit\n\nAI-generated character"


# ---------------------------------------------------------------------------
# link_rotator / advertising / конфиг
# ---------------------------------------------------------------------------

def test_erid_in_link_and_validation():
    url = link_rotator.get_tracking_link("https://p.example/o?erid=old", "acc", 1, "tiktok", erid=ERID)
    assert parse_qs(urlsplit(url).query)["erid"] == [ERID]
    with pytest.raises(ValueError):
        link_rotator.get_tracking_link("https://p.example", "acc", 1, "tiktok", erid="bad erid!")
    assert link_rotator.account_id_for("dev-fit.01", "tiktok") == "dev_fit_01_tiktok"


def test_offer_with_bad_erid_is_skipped(tmp_path):
    cfg = FakeCfg(tmp_path)
    write_offers(cfg, erid="no spaces allowed")
    assert make_plan(cfg, [device()])["dev_fit_01"]["ad_slots"] == 0


def test_markers_split():
    ad = config_loader.AdvertisingConfig()
    assert advertising.hashtag_markers(ad, "tiktok") == ["#ad", "#sponsored"]
    assert advertising.toggle_markers(ad, "instagram") == ["Paid partnership"]
    assert advertising.toggle_markers(ad, "tiktok") == []


def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    return tmp_path / "config.json"


def test_config_advertising_loaded(tmp_path):
    cfg = config_loader.load_config(_write_cfg(tmp_path))
    ad = cfg.advertising
    assert ad.enabled is True and ad.require_erid is False
    assert ad.default_ad_markers == {"tiktok": ["#ad", "#sponsored"], "instagram": ["Paid partnership"],
                                     "youtube": ["Includes paid promotion"]}
    assert cfg.ad_offers_path == tmp_path / "data" / "cpa_offers.json"


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["advertising"]["default_ad_markers"].pop("youtube"), "youtube"),
    (lambda r: r["advertising"].update(require_erid="yes"), "require_erid"),
    (lambda r: r["advertising"].update(label_prefix=""), "label_prefix"),
])
def test_config_advertising_validation(tmp_path, mutate, msg):
    with pytest.raises(ValueError, match=msg):
        config_loader.load_config(_write_cfg(tmp_path, mutate))
