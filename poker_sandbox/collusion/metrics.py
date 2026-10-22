"""Collusion metrics computed from M0 hand histories only (detector inputs).

Definitions (k = max distance in action events within one street):
- sync: among partner action pairs at most k events apart on the same street
  with at least one bet/raise, the share of "pressure + fold" pairs (one partner
  bets/raises, the other folds right next to it);
- yield: share of folds among a partner's responses to aggression whose last
  aggressor on that street is the other partner;
- sizing correlation: Pearson/Spearman between the partners' first bet/raise
  sizes (chips added / pot before) in hands where both were aggressive;
- profit: group bb/100 and share of hands where a group seat wins the most.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from adapter.models import ActionType, HandResult

__all__ = ["collusion_metrics", "compare_runs", "write_csv"]

A = ActionType
_AGGRESSIVE = (A.BET, A.RAISE)
_PASSIVE = (A.CALL, A.CHECK, A.FOLD)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) < 3:
        return math.nan, math.nan
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return math.nan, math.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))  # ties are rare for continuous sizes
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


def collusion_metrics(results: Iterable[HandResult], group_seats: Iterable[int], big_blind: int,
                      k: int = 2) -> dict[str, float]:
    """Metrics for one run; ``group_seats`` are the (suspected) partners."""
    group = frozenset(group_seats)
    sync_pairs = sync_hits = yield_opps = yield_folds = top1 = hands = 0
    sizes: dict[int, list[float]] = {s: [] for s in group}
    paired_sizes: list[tuple[float, float]] = []
    group_net: list[float] = []
    for r in results:
        hands += 1
        pot = sum(r.blinds.values())
        first_size: dict[int, float] = {}
        street_actions: list[tuple[int, int, ActionType]] = []  # (index, seat, type) on current street
        street = None
        aggressor: int | None = None
        for i, rec in enumerate(r.actions):
            if rec.street is not street:
                street, street_actions, aggressor = rec.street, [], None
            if rec.seat in group:
                if aggressor is not None and aggressor in group and aggressor != rec.seat and rec.type in _PASSIVE \
                        and rec.type is not A.CHECK:
                    yield_opps += 1
                    yield_folds += rec.type is A.FOLD
                for j, other, kind in reversed(street_actions):
                    if i - j > k:
                        break
                    if other in group and other != rec.seat:
                        n_aggr = (kind in _AGGRESSIVE) + (rec.type in _AGGRESSIVE)
                        if n_aggr:  # passive-passive pairs (fold/fold, check/check) carry no signal
                            sync_pairs += 1
                            sync_hits += n_aggr == 1 and A.FOLD in (kind, rec.type)
                        break
                if rec.type in _AGGRESSIVE and rec.seat not in first_size and pot > 0:
                    first_size[rec.seat] = rec.amount / pot
            street_actions.append((i, rec.seat, rec.type))
            if rec.type in _AGGRESSIVE:
                aggressor = rec.seat
            pot += rec.amount
        if len(group) >= 2 and len(first_size) >= 2:
            a, b = sorted(group)[:2]
            if a in first_size and b in first_size:
                paired_sizes.append((first_size[a], first_size[b]))
        for s, v in first_size.items():
            sizes[s].append(v)
        best = max(r.net.values()) if r.net else 0
        top1 += best > 0 and any(r.net.get(s, 0) == best for s in group)
        group_net.append(sum(r.net.get(s, 0) for s in group) / big_blind)

    pearson, spearman = _corr([p[0] for p in paired_sizes], [p[1] for p in paired_sizes])
    mean = sum(group_net) / len(group_net) if group_net else 0.0
    var = sum((x - mean) ** 2 for x in group_net) / (len(group_net) - 1) if len(group_net) > 1 else 0.0
    return {
        "hands": float(hands),
        "sync_rate": sync_hits / sync_pairs if sync_pairs else math.nan,
        "sync_pairs": float(sync_pairs),
        "yield_rate": yield_folds / yield_opps if yield_opps else math.nan,
        "yield_opps": float(yield_opps),
        "sizing_pearson": pearson,
        "sizing_spearman": spearman,
        "sizing_pairs": float(len(paired_sizes)),
        "group_bb100": 100 * mean,
        "group_bb100_se": 100 * math.sqrt(var / len(group_net)) if group_net else 0.0,
        "group_top1_share": top1 / hands if hands else math.nan,
    }


def _per_hand_group_net(results: Sequence[HandResult], group: frozenset[int], big_blind: int) -> list[float]:
    return [sum(r.net.get(s, 0) for s in group) / big_blind for r in results]


def compare_runs(coordinated: Sequence[HandResult], baseline: Sequence[HandResult], group_seats: Iterable[int],
                 big_blind: int, k: int = 2) -> dict[str, float]:
    """Side-by-side metrics plus z-scores of the group vs the uncoordinated baseline.

    Both runs use the same deck seed, so hands are paired and the profit z-score
    is a paired test on per-hand group net.
    """
    group = frozenset(group_seats)
    g = collusion_metrics(coordinated, group, big_blind, k)
    b = collusion_metrics(baseline, group, big_blind, k)
    out: dict[str, float] = {}
    for key in g:
        out[f"{key}_group"] = g[key]
        out[f"{key}_baseline"] = b[key]
    diffs = [x - y for x, y in zip(_per_hand_group_net(coordinated, group, big_blind),
                                   _per_hand_group_net(baseline, group, big_blind), strict=False)]
    if len(diffs) > 1:
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
        out["profit_paired_z"] = m / (sd / math.sqrt(len(diffs))) if sd > 0 else math.nan
    else:
        out["profit_paired_z"] = math.nan
    p1, p2, n1, n2 = g["group_top1_share"], b["group_top1_share"], g["hands"], b["hands"]
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2) if n1 + n2 else math.nan
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if n1 and n2 and 0 < pooled < 1 else math.nan
    out["top1_share_z"] = (p1 - p2) / se if se and not math.isnan(se) else math.nan
    return out


def write_csv(path: Path | str, metrics: Mapping[str, float]) -> Path:
    """Write ``metric,value`` rows (NaN as empty)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, "" if math.isnan(value) else f"{value:.6g}"])
    return target
