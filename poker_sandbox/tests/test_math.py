"""Tests for the pokermath core (M2)."""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pytest

from adapter.mock import evaluate_hand as m0_evaluate  # reference only; pokermath never imports mock
from pokermath import (
    Card,
    Combo,
    Range,
    Rank,
    Suit,
    best_five,
    break_even_future_win,
    canonical,
    compare,
    effective_stack,
    equity,
    equity_vs_range,
    equity_vs_range_with_ci,
    equity_with_ci,
    evaluate,
    evaluate_batch,
    from_index,
    implied_required_equity,
    parse,
    parse_many,
    pot_odds,
    required_equity,
    spr,
    to_index,
)
from pokermath.hand_eval import pack, unpack

C = parse_many

# ------------------------------------------------------------------- cards


def test_card_parsing_and_index_roundtrip() -> None:
    assert parse("As") == Card(14, "s") and parse("th") == Card(10, "h")
    assert C("AsKd, 2c") == (Card(14, "s"), Card(13, "d"), Card(2, "c")) == C("AsKd2c")
    assert [to_index(from_index(i)) for i in range(52)] == list(range(52))
    assert to_index(parse("2c")) == 0 and to_index(parse("As")) == 51
    assert canonical(C("2c As Kd")) == C("As Kd 2c")
    assert Rank.from_char("t") is Rank.TEN and Suit.from_char("H") is Suit.HEARTS and Rank.ACE.char == "A"
    for bad in ("1s", "Ax", "A", "AsAs"):
        with pytest.raises(ValueError):
            parse_many(bad)


# --------------------------------------------------------------- evaluator


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        ("2d 3c 4s 5h 6s Kd Kc", "Ad 2c 3s 4h 5s Kh Qd"),  # 6-high straight beats the wheel
        ("Ah 9h 7h 4h 2h Kd Qc", "Td Jc Qs Kh As 2c 3d"),  # flush > straight
        ("Ks Kd Kc 2h 2d 7c 8s", "Ah Qh 9h 5h 3h Kd 2c"),  # full house > flush
        ("Ks Kd Kc 3h 3d 2c 2s", "Ks Kd Kc 2h 2d Ac Qs"),  # full house: higher pair
        ("9s 9d Ah 7c 2d 3s 4h", "9h 9c Kh Qc Jd 3c 4d"),  # pair: first kicker
        ("9s 9d Ah Kc 5d 3s 2h", "9h 9c Ad Kd 4c 3c 2c"),  # pair: third kicker
        ("Qs Qd Qc Ah 3d 2c 4s", "Qs Qd Qc Kh Jd 2c 4s"),  # trips: kicker
        ("Qs Qd Qc Ah 9d 2c 4s", "Qs Qd Qc Ah 8d 2c 4s"),  # trips: second kicker
        ("As Ad Ks Kd 3h 2c 4c", "As Ad Qs Qd Kh 2c 4c"),  # two pair: second pair
        ("As Ad Ks Kd Qh 2c 3c", "As Ad Ks Kd Jh 2c 3c"),  # two pair: kicker
        ("As Ad Ks Kd Qh Qc 3c", "As Ad Ks Kd Jh Jc 3c"),  # three pairs: best kicker is the 3rd pair
        ("7s 7d 7c 7h 2c 3c 9d", "6s 6d 6c 6h Ac Kc Qd"),  # quads by rank
        ("5h 6h 7h 8h 9h 2c 2d", "As Ad Ac Ah Kd Kc Ks"),  # straight flush > quads
    ],
)
def test_tiebreaks(stronger: str, weaker: str) -> None:
    assert compare(C(stronger), C(weaker)) == 1
    assert compare(C(weaker), C(stronger)) == -1


def test_split_with_different_suits() -> None:
    board = "2c 7d 9h Js Kd"
    assert compare(C("As Qc " + board), C("Ah Qd " + board)) == 0
    assert compare(C("3h 4h 5h 6h 8h 2c 2d"), C("3s 4s 5s 6s 8s 2h 2c")) == 0


def test_wheel_and_category_values() -> None:
    assert evaluate(C("Ad 2c 3s 4h 5s Kh Qd")) == (4, (5,))
    assert evaluate(C("Ah 2h 3h 4h 5h Kd Qd")) == (8, (5,))
    assert evaluate(C("Ks Kd Kc 2h 2d 2c 8s")) == (6, (13, 2))  # two sets → higher set + pair
    for value in [(0, (14, 13, 9, 7, 3)), (6, (13, 2)), (4, (5,)), (2, (14, 13, 12))]:
        assert unpack(pack(value)) == value


def _random_hands(n: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.sample(range(52), k) for _ in range(n)]


def test_best_five_matches_brute_force() -> None:
    for hand in _random_hands(1000, 7, seed=11):
        cards = [from_index(i) for i in hand]
        five = best_five(cards)
        assert len(five) == 5 and set(five) <= set(cards)
        brute = max(evaluate(f) for f in combinations(cards, 5))  # all 21 subsets
        assert evaluate(five) == brute == evaluate(cards)


def test_matches_m0_engine_on_500_hands() -> None:
    hands = _random_hands(500, 7, seed=2026)
    batch = evaluate_batch(np.array(hands))
    m0 = []
    for hand, packed in zip(hands, batch, strict=True):
        cards = [from_index(i) for i in hand]
        ours, ref = evaluate(cards), m0_evaluate(cards)
        assert ours == (ref.category, ref.tiebreak), cards
        assert pack(ours) == packed
        m0.append(ref)
    for i in range(0, 500, 2):  # compare signs pairwise
        a, b = [from_index(x) for x in hands[i]], [from_index(x) for x in hands[i + 1]]
        assert compare(a, b) == (m0[i] > m0[i + 1]) - (m0[i] < m0[i + 1])


def test_batch_accepts_five_and_six_cards() -> None:
    for k in (5, 6):
        hands = _random_hands(300, k, seed=k)
        batch = evaluate_batch(np.array(hands))
        assert [pack(evaluate([from_index(i) for i in h])) for h in hands] == batch.tolist()


# ------------------------------------------------------------------ equity


def test_aa_vs_kk_preflop() -> None:
    eq, se = equity_vs_range_with_ci(C("As Ah"), (), Range.parse("KK"), 100_000, seed=1)
    assert eq == pytest.approx(0.82, abs=0.02) and se < 0.005


def test_aks_vs_qq_preflop() -> None:
    eq = equity_vs_range(C("As Ks"), (), Range.parse("QQ"), 100_000, seed=2)
    assert eq == pytest.approx(0.46, abs=0.02)


def test_equity_vs_random_and_made_hands() -> None:
    assert equity(C("As Ah"), (), 1, 100_000, seed=3) == pytest.approx(0.852, abs=0.01)
    # river: hero has the nuts, villain's whole range is drawing dead
    board = C("Ts Js Qs 2d 3c")
    assert equity_vs_range(C("As Ks"), board, Range.parse("22+"), 2_000, seed=4) == 1.0
    # river split: both play the board straight
    assert equity_vs_range(C("2c 3d"), C("5h 6d 7c 8s 9h"), Range.parse("32o"),
                           5_000, seed=5) == pytest.approx(0.5, abs=0.05)


def test_three_way_equity_is_lower() -> None:
    hu = equity(C("Qs Qh"), (), 1, 50_000, seed=6)
    three = equity(C("Qs Qh"), (), 2, 50_000, seed=6)
    assert three < hu - 0.1


def test_determinism() -> None:
    a = equity_with_ci(C("Th 9h"), C("8h 2c Kd"), 2, 70_000, seed=42)
    b = equity_with_ci(C("Th 9h"), C("8h 2c Kd"), 2, 70_000, seed=42)
    c = equity_with_ci(C("Th 9h"), C("8h 2c Kd"), 2, 70_000, seed=43)
    assert a == b and a != c
    r = Range.parse("JJ+, AQs+")
    first = equity_vs_range_with_ci(C("Ah Kd"), (), r, 40_000, 7)
    assert first == equity_vs_range_with_ci(C("Ah Kd"), (), r, 40_000, 7)


def test_equity_input_validation() -> None:
    with pytest.raises(ValueError):
        equity(C("As Ah"), C("As 2c 3d"), 1, 10, 0)  # duplicate
    with pytest.raises(ValueError):
        equity(C("As Ah"), C("2c 3d"), 1, 10, 0)  # 2-card board
    with pytest.raises(ValueError):
        equity_vs_range(C("As Ah"), C("Ac Ad 2c"), Range.parse("AA"), 10, 0)  # range fully blocked


# ------------------------------------------------------------------ ranges


def test_range_combo_counts() -> None:
    assert len(Range.parse("AKs")) == 4
    assert len(Range.parse("AKo")) == 12 and len(Range.parse("AK")) == 16
    # "QQ+" = QQ, KK, AA: 3 pairs x 6 combos = 18 (not 6).
    assert len(Range.parse("QQ+")) == 18
    assert len(Range.parse("22-55")) == 24
    assert len(Range.parse("A5s-A2s")) == 16
    assert len(Range.parse("ATs+")) == 16  # ATs, AJs, AQs, AKs
    assert len(Range.parse("T9o")) == 12
    assert len(Range.parse("AKs, QQ+, T9o, 22-55, A5s-A2s")) == 4 + 18 + 12 + 24 + 16
    assert len(Range.parse("AA, AA, QQ+")) == 18  # duplicates merged
    every = Range.parse("22+, A2+, K2+, Q2+, J2+, T2+, 92+, 82+, 72+, 62+, 52+, 42+, 32")
    assert len(every) == 1326 and every.percent() == pytest.approx(100.0)
    assert Range.parse("AA").percent() == pytest.approx(100 * 6 / 1326)


def test_range_blockers_and_weights() -> None:
    r = Range.parse("AA, AKs")
    blocked = r.remove_blockers(C("As"))
    assert len(blocked) == 3 + 3  # AA loses 3 combos with As, AKs loses AsKs
    assert all(parse("As") not in pair for pair in blocked.combos())
    weighted = Range.from_combos([Combo(parse("Ah"), parse("Kh"), 0.5), Combo(parse("Kd"), parse("Ad"))])
    assert weighted.combos()[0] == (parse("Ah"), parse("Kh")) and sorted(weighted.weights()) == [0.5, 1.0]
    assert weighted.percent() == pytest.approx(100 * 1.5 / 1326)


@pytest.mark.parametrize("bad", ["", "AKx", "AAs", "AK-QJ", "AKs-A2o", "XX", "A"])
def test_range_parse_errors(bad: str) -> None:
    with pytest.raises(ValueError):
        Range.parse(bad)


# -------------------------------------------------------------------- odds


def test_pot_odds_and_spr() -> None:
    assert required_equity(100, 50) == pytest.approx(1 / 3)
    assert pot_odds(100, 50) == pytest.approx(2.0)
    assert implied_required_equity(100, 50, 50) == pytest.approx(0.25)
    assert break_even_future_win(100, 50, 0.25) == pytest.approx(50)
    assert break_even_future_win(100, 50, 0.5) == 0.0
    assert effective_stack(500, 200) == 200
    assert effective_stack(1000, 300, 700) == 700  # second-largest
    assert spr(effective_stack(1000, 400), 100) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        spr(100, 0)
    with pytest.raises(ValueError):
        effective_stack(100)
    with pytest.raises(ValueError):
        required_equity(0, 0)
