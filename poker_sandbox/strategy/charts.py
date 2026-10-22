"""Preflop charts for 6-max NLHE and helpers over the 169 starting-hand classes.

Source: standard 6-max ranges — plausible, not optimal. Strings use the
``pokermath.ranges`` notation; ``raise_bluff`` hands are raised only sometimes
(see ``HeuristicParams.bluff_freq``), everything else is weight 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

from adapter.models import RANK_CHARS, Card
from pokermath.ranges import TOTAL_COMBOS, Combo, Range


class Position(StrEnum):
    UTG = "UTG"
    MP = "MP"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"


class Spot(StrEnum):
    OPEN = "open"  # unopened pot (limpers allowed); BB uses it to raise limpers
    VS_OPEN = "vs_open"  # facing one raise, not in the BB
    VS_3BET = "vs_3bet"  # facing a re-raise (3-bet or more)
    BB_DEFEND = "bb_defend"  # BB facing one raise


@dataclass(frozen=True, slots=True)
class SpotChart:
    """Raise-for-value, raise-as-bluff and flat-call ranges for one spot."""

    raise_value: str
    raise_bluff: str = ""
    call: str = ""


# Source: standard 6-max ranges, plausible, not optimal.
OPEN_CHARTS: dict[Position, SpotChart] = {
    Position.UTG: SpotChart("22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo"),
    Position.MP: SpotChart("22+, A2s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KJo+, QJo"),
    Position.CO: SpotChart("22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 76s, 65s, A8o+, KTo+, QTo+, JTo"),
    Position.BTN: SpotChart(
        "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K8o+, Q9o+, J9o+, T9o, 98o"
    ),
    Position.SB: SpotChart("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, 65s, A5o+, K9o+, QTo+, JTo"),
    Position.BB: SpotChart("TT+, AQs+, AKo"),  # raise over limpers
}

VS_OPEN_CHARTS: dict[Position, SpotChart] = {
    Position.UTG: SpotChart("QQ+, AKs, AKo", "", "JJ-99, AQs, KQs"),  # only vs a straddle-like open
    Position.MP: SpotChart("QQ+, AKs, AKo", "A5s-A4s", "JJ-77, AQs, AJs, KQs, QJs, JTs"),
    Position.CO: SpotChart("QQ+, AKs, AKo", "A5s-A3s, K9s", "JJ-66, AQs-ATs, KQs, KJs, QJs, JTs, T9s, AQo"),
    Position.BTN: SpotChart(
        "JJ+, AKs, AKo", "A5s-A2s, K9s, 76s",
        "TT-22, AQs-A9s, KQs-KTs, QJs, QTs, JTs, T9s, 98s, 87s, AQo, AJo, KQo",
    ),
    Position.SB: SpotChart("QQ+, AKs, AKo", "A5s-A4s, KTs", "JJ-88, AQs, AJs, KQs"),
}

BB_DEFEND_CHART = SpotChart(
    "QQ+, AKs, AKo",
    "A5s-A2s, K9s, Q9s, 65s",
    "JJ-22, AQs-A6s, K2s+, Q8s+, J8s+, T7s+, 97s+, 86s+, 75s+, 64s+, 54s, AQo-A8o, KTo+, QTo+, JTo",
)

# SB vs a BTN open: wider, mostly 3-bet-or-fold.
SB_VS_BTN_CHART = SpotChart(
    "TT+, AQs+, AKo", "A9s-A2s, KTs+, QTs+, JTs, 98s", "99-66, AJs, ATs, KQo, AJo"
)

VS_3BET_CHARTS: dict[Position, SpotChart] = {
    Position.UTG: SpotChart("KK+, AKs", "", "QQ-TT, AKo, AQs, KQs"),
    Position.MP: SpotChart("KK+, AKs", "A5s", "QQ-99, AKo, AQs, KQs"),
    Position.CO: SpotChart("QQ+, AKs, AKo", "A5s-A4s", "JJ-77, AQs-ATs, KQs, QJs, JTs, AQo"),
    Position.BTN: SpotChart("QQ+, AKs, AKo", "A5s-A3s", "JJ-66, AQs-A9s, KQs, KJs, QJs, JTs, T9s, AQo, AJo"),
    Position.SB: SpotChart("QQ+, AKs, AKo", "A5s-A4s", "JJ-88, AQs-ATs, KQs, AQo"),
    Position.BB: SpotChart("QQ+, AKs, AKo", "A5s", "JJ-88, AQs, AJs, KQs"),
}

# Facing a 4-bet or more: jam or fold.
VS_4BET_CHART = SpotChart("QQ+, AKs, AKo")


def spot_chart(position: Position, spot: Spot, opener: Position | None = None) -> SpotChart:
    """The chart for ``position`` in ``spot``; SB facing a BTN open uses the wider SB-vs-BTN chart."""
    if spot is Spot.OPEN:
        return OPEN_CHARTS[position]
    if spot is Spot.BB_DEFEND or (spot is Spot.VS_OPEN and position is Position.BB):
        return BB_DEFEND_CHART
    if spot is Spot.VS_OPEN:
        if position is Position.SB and opener is Position.BTN:
            return SB_VS_BTN_CHART
        return VS_OPEN_CHARTS[position]
    return VS_3BET_CHARTS[position]


def chart_for(position: Position, spot: Spot, opener: Position | None = None) -> Range:
    """Every hand that continues in the spot (raise value + raise bluff + call), weight 1.0."""
    chart = spot_chart(position, spot, opener)
    return parse_cached(", ".join(s for s in (chart.raise_value, chart.raise_bluff, chart.call) if s))


# ------------------------------------------------------------ hand classes


def hand_class(c1: Card, c2: Card) -> str:
    """``"AKs"``, ``"AKo"`` or ``"AA"`` for two hole cards."""
    hi, lo = (c1, c2) if c1.rank >= c2.rank else (c2, c1)
    label = RANK_CHARS[hi.rank - 2] + RANK_CHARS[lo.rank - 2]
    if hi.rank == lo.rank:
        return label
    return label + ("s" if hi.suit == lo.suit else "o")


def class_combos(label: str) -> int:
    """Number of card combos in a class: 6 pairs, 4 suited, 12 offsuit."""
    return 6 if len(label) == 2 else (4 if label[2] == "s" else 12)


def chen_score(label: str) -> float:
    """Chen formula, used only to order classes from strongest to weakest."""
    values = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0}
    r1, r2 = RANK_CHARS.index(label[0]) + 2, RANK_CHARS.index(label[1]) + 2
    high = values.get(label[0], r1 / 2)
    if r1 == r2:
        return max(5.0, high * 2)
    score = high + (2.0 if label.endswith("s") else 0.0)
    gap = r1 - r2 - 1
    score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and r1 < 12:
        score += 1
    return math.ceil(score)


@cache
def all_classes() -> tuple[str, ...]:
    """All 169 classes, strongest first by Chen score (ties: more combos-efficient label order)."""
    labels = []
    for i, a in enumerate(reversed(RANK_CHARS)):
        for b in reversed(RANK_CHARS[: len(RANK_CHARS) - i]):
            if a == b:
                labels.append(a + b)
            else:
                labels += [a + b + "s", a + b + "o"]
    return tuple(sorted(labels, key=lambda lab: (-chen_score(lab), lab)))


@cache
def parse_cached(text: str) -> Range:
    """``Range.parse`` with memoisation (charts are parsed once)."""
    return Range.parse(text)


@cache
def classes_of(text: str) -> frozenset[str]:
    """Hand classes fully contained in a chart string."""
    if not text:
        return frozenset()
    counts: dict[str, int] = {}
    for c1, c2 in parse_cached(text).combos():
        label = hand_class(c1, c2)
        counts[label] = counts.get(label, 0) + 1
    return frozenset(lab for lab, n in counts.items() if n == class_combos(lab))


def scale_classes(classes: Iterable[str], factor: float) -> dict[str, float]:
    """Grow/shrink a class set to ``factor`` times its combos along the Chen order.

    Returns class → weight; the boundary class gets a fractional weight so the
    combo count matches exactly (the caller mixes on it with its RNG).
    """
    base = set(classes)
    if not base:
        return {}
    current = sum(class_combos(c) for c in base)
    target = min(float(TOTAL_COMBOS), current * factor)
    order = all_classes()
    if factor >= 1:
        out = dict.fromkeys(base, 1.0)
        extra = target - current
        for label in order:
            if extra <= 0:
                break
            if label in out:
                continue
            n = class_combos(label)
            out[label] = min(1.0, extra / n)
            extra -= n
        return out
    out = dict.fromkeys(base, 1.0)
    excess = current - target
    for label in reversed(order):  # weakest first
        if excess <= 0:
            break
        if label not in out:
            continue
        n = class_combos(label)
        if n <= excess:
            del out[label]
        else:
            out[label] = 1.0 - excess / n
        excess -= n
    return out


def classes_to_range(weights: Mapping[str, float]) -> Range:
    """Expand class weights into a weighted :class:`Range`."""
    combos = []
    for label, weight in weights.items():
        if weight <= 0:
            continue
        for c1, c2 in parse_cached(label).combos():
            combos.append(Combo(c1, c2, min(1.0, weight)))
    return Range.from_combos(combos)
