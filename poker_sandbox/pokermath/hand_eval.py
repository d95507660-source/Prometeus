"""Best-5-of-7 hand evaluation: a scalar path and a vectorised numpy path with one format.

Result convention matches the M0 engine: ``(category, tiebreak)`` with categories
0 = high card … 8 = straight flush and tiebreak ranks 2..14 ordered by group
size then rank (e.g. full house = (trips, pair)); straights use the top rank (wheel = 5).

The packed form is ``category << 20 | tiebreak nibbles`` (first tiebreak in bits
16..19, unused low nibbles are 0). Within a category every tiebreak has the same
length, so packed values compare exactly like ``(category, tiebreak)`` tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from adapter.models import Card
from pokermath.cards import to_index

__all__ = [
    "CATEGORY_NAMES",
    "HandValue",
    "best_five",
    "compare",
    "evaluate",
    "evaluate_batch",
    "evaluate_indices",
    "pack",
    "unpack",
]

CATEGORY_NAMES: Final = (
    "High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
)
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

HandValue = tuple[int, tuple[int, ...]]

_NMASKS: Final = 1 << 13
_WHEEL: Final = 0b1_0000_0000_1111  # A,5,4,3,2


def _build_tables() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per 13-bit rank mask: straight top rank (0 = none), top-5 ranks packed in nibbles, popcount."""
    straight = np.zeros(_NMASKS, dtype=np.int64)
    top5 = np.zeros(_NMASKS, dtype=np.int64)
    popcnt = np.zeros(_NMASKS, dtype=np.int64)
    for mask in range(_NMASKS):
        ranks = [r for r in range(12, -1, -1) if mask >> r & 1]  # rank_idx desc
        popcnt[mask] = len(ranks)
        packed = 0
        for i, r in enumerate(ranks[:5]):
            packed |= (r + 2) << (4 * (4 - i))
        top5[mask] = packed
        for hi in range(12, 3, -1):
            if (mask >> (hi - 4)) & 0b11111 == 0b11111:
                straight[mask] = hi + 2
                break
        else:
            if mask & _WHEEL == _WHEEL:
                straight[mask] = 5
    return straight, top5, popcnt


STRAIGHT_TOP, TOP5, POPCOUNT = _build_tables()
_STRAIGHT_L: Final = STRAIGHT_TOP.tolist()
_TOP5_L: Final = TOP5.tolist()
_RANK_BITS: Final = (1 << np.arange(13, dtype=np.int64))


def _top(mask: int, k: int) -> tuple[int, ...]:
    packed = _TOP5_L[mask]
    return tuple((packed >> (4 * (4 - i))) & 0xF for i in range(k))


def pack(value: HandValue) -> int:
    """``(category, tiebreak)`` → int (same order)."""
    category, tiebreak = value
    packed = category << 20
    for i, r in enumerate(tiebreak):
        packed |= r << (4 * (4 - i))
    return packed


_TB_LEN: Final = (5, 4, 3, 3, 1, 5, 2, 2, 1)


def unpack(packed: int) -> HandValue:
    """Inverse of :func:`pack`."""
    category = packed >> 20
    return category, tuple((packed >> (4 * (4 - i))) & 0xF for i in range(_TB_LEN[category]))


def evaluate_indices(idx: Sequence[int]) -> HandValue:
    """Scalar evaluation of 5..7 card indices (0..51)."""
    if not 5 <= len(idx) <= 7 or len(set(idx)) != len(idx):
        raise ValueError("need 5 to 7 distinct cards")
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    for c in idx:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r
    rank_mask = pair_mask = trip_mask = quad_mask = 0
    for r, n in enumerate(counts):
        bit = 1 << r
        if n:
            rank_mask |= bit
        if n >= 2:
            pair_mask |= bit
        if n >= 3:
            trip_mask |= bit
        if n == 4:
            quad_mask |= bit
    flush_mask = next((m for m in suit_masks if m.bit_count() >= 5), 0)

    if flush_mask and (sf := _STRAIGHT_L[flush_mask]):
        return STRAIGHT_FLUSH, (sf,)
    if quad_mask:
        q = _top(quad_mask, 1)[0]
        return QUADS, (q, _top(rank_mask & ~(1 << (q - 2)), 1)[0])
    if trip_mask:
        t = _top(trip_mask, 1)[0]
        rest_pairs = pair_mask & ~(1 << (t - 2))  # a second set also plays as the pair
        if rest_pairs:
            return FULL_HOUSE, (t, _top(rest_pairs, 1)[0])
    if flush_mask:
        return FLUSH, _top(flush_mask, 5)
    if st := _STRAIGHT_L[rank_mask]:
        return STRAIGHT, (st,)
    if trip_mask:
        t = _top(trip_mask, 1)[0]
        return TRIPS, (t, *_top(rank_mask & ~(1 << (t - 2)), 2))
    if pair_mask.bit_count() >= 2:
        hi, lo = _top(pair_mask, 2)
        return TWO_PAIR, (hi, lo, _top(rank_mask & ~(1 << (hi - 2)) & ~(1 << (lo - 2)), 1)[0])
    if pair_mask:
        p = _top(pair_mask, 1)[0]
        return PAIR, (p, *_top(rank_mask & ~(1 << (p - 2)), 3))
    return HIGH, _top(rank_mask, 5)


def evaluate(cards: Sequence[Card]) -> HandValue:
    """Best 5-of-(5..7) value as ``(category, tiebreak)``."""
    return evaluate_indices([to_index(c) for c in cards])


def compare(hand_a: Sequence[Card], hand_b: Sequence[Card]) -> int:
    """1 if ``hand_a`` is stronger, -1 if weaker, 0 on a split."""
    a, b = evaluate(hand_a), evaluate(hand_b)
    return (a > b) - (a < b)


# Cards contributed by each tiebreak entry, per category (straights handled separately).
_GROUPS: Final = {QUADS: (4, 1), FULL_HOUSE: (3, 2), TRIPS: (3, 1, 1), TWO_PAIR: (2, 2, 1),
                  PAIR: (2, 1, 1, 1), HIGH: (1, 1, 1, 1, 1)}


def best_five(cards7: Sequence[Card]) -> tuple[Card, ...]:
    """The five cards forming the best hand (ties between equal suits broken by suit order)."""
    category, tiebreak = evaluate(cards7)
    by_strength = sorted(cards7, key=to_index, reverse=True)
    if category in (STRAIGHT_FLUSH, FLUSH):
        suit = next(s for s in "cdhs" if sum(c.suit == s for c in cards7) >= 5)
        pool = [c for c in by_strength if c.suit == suit]
    else:
        pool = by_strength
    if category in (STRAIGHT_FLUSH, STRAIGHT):
        top = tiebreak[0]
        wanted = [14 if r == 1 else r for r in range(top, top - 5, -1)]
        return tuple(next(c for c in pool if c.rank == r) for r in wanted)
    if category == FLUSH:
        return tuple(pool[:5])
    out: list[Card] = []
    for rank, n in zip(tiebreak, _GROUPS[category], strict=True):
        out.extend([c for c in pool if c.rank == rank][:n])
    return tuple(out)


def evaluate_batch(hands: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Vectorised evaluation: ``(N, 5..7)`` card indices → ``(N,)`` packed values."""
    idx = np.asarray(hands, dtype=np.int64)
    if idx.ndim != 2 or not 5 <= idx.shape[1] <= 7:
        raise ValueError("expected an (N, 5..7) array of card indices")
    ranks = idx >> 2
    bits = _RANK_BITS[ranks]  # (N, k)
    counts = (ranks[:, :, None] == np.arange(13)).sum(axis=1)  # (N, 13)
    rank_mask = (counts >= 1) @ _RANK_BITS
    pair_mask = (counts >= 2) @ _RANK_BITS
    trip_mask = (counts >= 3) @ _RANK_BITS
    quad_mask = (counts == 4) @ _RANK_BITS

    suits = idx & 3
    # Ranks within one suit are distinct, so a sum of rank bits equals their OR.
    suit_masks = np.stack([np.where(suits == s, bits, 0).sum(axis=1) for s in range(4)], axis=1)
    flushy = POPCOUNT[suit_masks] >= 5  # at most one suit can have 5+ of 7 cards
    flush_mask = np.where(flushy, suit_masks, 0).max(axis=1)
    has_flush = flush_mask > 0

    def top(mask: npt.NDArray[np.int64], k: int) -> npt.NDArray[np.int64]:
        return TOP5[mask] >> (4 * (5 - k))  # top-k ranks, already left-aligned

    def bit(rank_nibble: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.where(rank_nibble >= 2, np.left_shift(1, np.maximum(rank_nibble - 2, 0)), 0)

    sf_top = np.where(has_flush, STRAIGHT_TOP[flush_mask], 0)
    st_top = STRAIGHT_TOP[rank_mask]
    q = top(quad_mask, 1)
    t = top(trip_mask, 1)
    p2 = top(pair_mask, 2)
    p_hi, p_lo = p2 >> 4, p2 & 0xF
    fh_pair = top(pair_mask & ~bit(t), 1)

    val_sf = (STRAIGHT_FLUSH << 20) | (sf_top << 16)
    val_quads = (QUADS << 20) | (q << 16) | (top(rank_mask & ~bit(q), 1) << 12)
    val_fh = (FULL_HOUSE << 20) | (t << 16) | (fh_pair << 12)
    val_flush = (FLUSH << 20) | TOP5[flush_mask]
    val_straight = (STRAIGHT << 20) | (st_top << 16)
    val_trips = (TRIPS << 20) | (t << 16) | (top(rank_mask & ~bit(t), 2) << 8)
    val_two = (TWO_PAIR << 20) | (p_hi << 16) | (p_lo << 12) | (top(rank_mask & ~bit(p_hi) & ~bit(p_lo), 1) << 8)
    val_pair = (PAIR << 20) | (p_hi << 16) | (top(rank_mask & ~bit(p_hi), 3) << 4)
    val_high = (HIGH << 20) | TOP5[rank_mask]

    conditions = [
        sf_top > 0,
        quad_mask > 0,
        (trip_mask > 0) & (fh_pair > 0),
        has_flush,
        st_top > 0,
        trip_mask > 0,
        POPCOUNT[pair_mask] >= 2,
        pair_mask > 0,
    ]
    choices = [val_sf, val_quads, val_fh, val_flush, val_straight, val_trips, val_two, val_pair]
    result: npt.NDArray[np.int64] = np.select(conditions, choices, default=val_high).astype(np.int64)
    return result

