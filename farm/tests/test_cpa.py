"""tests/test_cpa.py — подбор оффера и UTM-ссылки (без сети).

Запуск:  python -m pytest -q tests/test_cpa.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpa_matcher import find_offer, load_offers, match_keywords  # noqa: E402
from link_rotator import get_tracking_link  # noqa: E402

OFFERS = [
    {"id": 1, "keywords": ["tracker", "habit"], "payout": 2.5, "url": "https://a.example/1"},
    {"id": 2, "keywords": ["habit"], "payout": 4.0, "url": "https://b.example/2"},
    {"id": 3, "keywords": ["budget app", "finance"], "payout": 6.0, "url": "https://c.example/3"},
    {"id": 4, "keywords": ["habit"], "payout": 9.0, "url": "https://d.example/4", "enabled": False},
    {"id": 5, "keywords": ["трекер", "привычки"], "payout": 3.0, "url": "https://e.example/5"},
]


@pytest.fixture
def offers_file(tmp_path: Path) -> Path:
    f = tmp_path / "cpa_offers.json"
    f.write_text(json.dumps(OFFERS, ensure_ascii=False), encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# cpa_matcher
# ---------------------------------------------------------------------------

def test_best_payout_wins(offers_file):
    idea = {"topic": "Build a daily habit in 5 minutes", "key_points": ["use a tracker"]}
    best = find_offer(idea, str(offers_file))
    assert best["id"] == 2 and best["payout"] == 4.0  # 1 и 2 подходят, у 2 payout больше
    assert best["matched_keywords"] == ["habit"]


def test_disabled_offer_ignored(offers_file):
    best = find_offer({"topic": "habit"}, offers_file)
    assert best["id"] != 4


def test_match_in_key_points_only(offers_file):
    idea = {"topic": "Morning routine", "key_points": ["Track your spending with a budget app"]}
    assert find_offer(idea, offers_file)["id"] == 3


def test_whole_word_and_plural(offers_file):
    assert match_keywords({"topic": "best trackers of 2026"}, ["tracker"]) == ["tracker"]
    assert match_keywords({"topic": "habits that stick"}, ["habit"]) == ["habit"]
    assert match_keywords({"topic": "inhabitant stories"}, ["habit"]) == []
    assert match_keywords({"key_points": ["Budget   App review"]}, ["budget app"]) == ["budget app"]
    assert match_keywords({"hashtags": ["#habit_tracker"]}, ["tracker"]) == ["tracker"]


def test_cyrillic_keywords(offers_file):
    assert find_offer({"topic": "Лучший трекер привычки на каждый день"}, offers_file)["id"] == 5


def test_no_match_returns_none(offers_file, caplog):
    with caplog.at_level(logging.INFO, logger="cpa_matcher"):
        assert find_offer({"topic": "cooking pasta"}, offers_file) is None
    assert any("не найден" in r.getMessage() for r in caplog.records)


def test_tie_break_by_matches_then_id(tmp_path):
    f = tmp_path / "o.json"
    f.write_text(json.dumps([
        {"id": 7, "keywords": ["habit"], "payout": 5, "url": "https://x.example"},
        {"id": 3, "keywords": ["habit"], "payout": 5, "url": "https://x.example"},
        {"id": 9, "keywords": ["habit", "tracker"], "payout": 5, "url": "https://x.example"},
    ]), encoding="utf-8")
    assert find_offer({"topic": "habit tracker"}, f)["id"] == 9  # больше совпадений
    assert find_offer({"topic": "habit"}, f)["id"] == 3          # затем меньший id


def test_missing_file_logged_returns_none(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="cpa_matcher"):
        assert find_offer({"topic": "habit"}, tmp_path / "nope.json") is None
    assert any("не найден" in r.getMessage() for r in caplog.records)


def test_broken_json_logged_returns_none(tmp_path, caplog):
    f = tmp_path / "bad.json"
    f.write_text("[{\"id\": 1,", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="cpa_matcher"):
        assert find_offer({"topic": "habit"}, f) is None
    assert any("битый JSON" in r.getMessage() for r in caplog.records)


def test_invalid_offers_skipped(tmp_path):
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"offers": [
        {"id": 1, "keywords": ["habit"], "payout": "a lot", "url": "https://x.example"},
        {"id": 2, "keywords": ["habit"], "payout": 1, "url": "javascript:alert(1)"},
        {"id": "bad id", "keywords": ["habit"], "payout": 1, "url": "https://x.example"},
        {"id": 4, "keywords": [], "payout": 1, "url": "https://x.example"},
        {"id": 5, "keywords": ["habit"], "payout": -1, "url": "https://x.example"},
        {"id": 6, "keywords": ["habit"], "payout": 1.5, "url": "https://ok.example"},
    ]}), encoding="utf-8")
    assert [o["id"] for o in load_offers(f)] == [6]
    assert find_offer({"topic": "habit"}, f)["id"] == 6


def test_non_dict_idea(offers_file):
    assert find_offer("habit", offers_file) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# link_rotator
# ---------------------------------------------------------------------------

def test_utm_link_format():
    url = get_tracking_link("https://partner.example/offer", "aria_fight", 7, "tiktok")
    assert url == ("https://partner.example/offer?utm_source=tiktok&utm_medium=video"
                   "&utm_campaign=account_aria_fight&utm_content=offer_7")


def test_existing_query_and_fragment_kept_utm_overwritten():
    url = get_tracking_link("https://p.example/o?ref=42&utm_source=old#top", "acc_1", 3, "youtube")
    parts = urlsplit(url)
    q = parse_qs(parts.query)
    assert q["ref"] == ["42"] and q["utm_source"] == ["youtube"] and parts.fragment == "top"
    assert url.count("utm_source") == 1


def test_links_unique_per_account_offer_platform():
    links = {get_tracking_link("https://p.example/", acc, off, plat)
             for acc in ("a1", "a2") for off in (1, 2) for plat in ("tiktok", "instagram")}
    assert len(links) == 8


def test_http_allowed_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="link_rotator"):
        url = get_tracking_link("http://p.example/x", "acc", 1, "tiktok")
    assert url.startswith("http://") and any("без https" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("base_url", [
    "ftp://p.example/x", "javascript:alert(1)", "//p.example/x", "https://", "p.example",
    "https://user:pass@p.example/x", "https://p.example/a b", "https://p.example/\nx", 123,
])
def test_bad_base_url_rejected(base_url):
    with pytest.raises(ValueError):
        get_tracking_link(base_url, "acc", 1, "tiktok")


@pytest.mark.parametrize("account_id", ["acc-1", "acc 1", "", "a/b", "acc&utm=1", "я", "x" * 65, None])
def test_bad_account_id_rejected(account_id):
    with pytest.raises(ValueError):
        get_tracking_link("https://p.example", account_id, 1, "tiktok")


@pytest.mark.parametrize("offer_id", [-1, True, "1;2", "", 1.5, None])
def test_bad_offer_id_rejected(offer_id):
    with pytest.raises(ValueError):
        get_tracking_link("https://p.example", "acc", offer_id, "tiktok")


def test_string_offer_id_allowed():
    assert get_tracking_link("https://p.example", "acc", "promo_2", "tiktok").endswith(
        "utm_content=offer_promo_2")


@pytest.mark.parametrize("platform", ["TikTok", "tik tok", "", "x&y=1"])
def test_bad_platform_rejected(platform):
    with pytest.raises(ValueError):
        get_tracking_link("https://p.example", "acc", 1, platform)
