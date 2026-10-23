"""Detector quality metrics over run records: detection rate, FPR, time-to-detect, matrix, seed CIs."""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.runner import RunRecord

__all__ = [
    "Stat",
    "Summary",
    "compute_summary",
    "detection_matrix",
    "detection_rate",
    "false_positive_rate",
    "mean_ci",
    "time_to_detect",
    "write_csv",
]

# two-sided 95 % Student t quantiles for df = 1..30 (no scipy); normal beyond
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


@dataclass(frozen=True, slots=True)
class Stat:
    """mean ± std (ddof=1) and 95 % t-interval over ``n`` values (per-seed aggregates)."""

    mean: float
    std: float
    lo: float
    hi: float
    n: int

    def fmt(self) -> str:
        """``0.600 ± 0.100 [0.476, 0.724]``."""
        if math.isnan(self.mean):
            return "n/a"
        return f"{self.mean:.3f} ± {self.std:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def mean_ci(values: Iterable[float]) -> Stat:
    """Aggregate per-seed values; NaNs are dropped, a single value has zero width."""
    xs = [v for v in values if not math.isnan(v)]
    if not xs:
        return Stat(math.nan, math.nan, math.nan, math.nan, 0)
    mean = statistics.fmean(xs)
    if len(xs) == 1:
        return Stat(mean, 0.0, mean, mean, 1)
    std = statistics.stdev(xs)
    t = _T95[len(xs) - 2] if len(xs) - 1 <= len(_T95) else 1.96
    half = t * std / math.sqrt(len(xs))
    return Stat(mean, std, mean - half, mean + half, len(xs))


def _rate(records: Iterable[RunRecord], threshold: float) -> float:
    ok = [r for r in records if r.ok]
    return sum(r.flagged(threshold) for r in ok) / len(ok) if ok else math.nan


def detection_rate(records: Iterable[RunRecord], threshold: float = 0.5) -> float:
    """Share of ok bot runs flagged (is_bot and confidence ≥ threshold)."""
    return _rate((r for r in records if r.label == "bot"), threshold)


def false_positive_rate(records: Iterable[RunRecord], threshold: float = 0.5) -> float:
    """Share of ok baseline runs flagged."""
    return _rate((r for r in records if r.label == "baseline"), threshold)


def time_to_detect(prefix: Sequence[Sequence[Any]], threshold: float = 0.5) -> int | None:
    """Smallest checkpoint from which every later checkpoint is flagged (stable, not the first hit)."""
    stable: int | None = None
    for n, is_bot, confidence in (row[:3] for row in prefix):
        hit = bool(is_bot) and confidence >= threshold
        stable = (int(n) if stable is None else stable) if hit else None  # any miss resets the streak
    return stable


def detection_matrix(records: Iterable[RunRecord], threshold: float = 0.5) -> dict[str, dict[str, float]]:
    """profile → detector → flagged share over seeds (insertion order follows the records)."""
    groups: dict[str, dict[str, list[RunRecord]]] = {}
    for r in records:
        groups.setdefault(r.profile_id, {}).setdefault(r.detector_id, []).append(r)
    return {p: {d: _rate(rs, threshold) for d, rs in ds.items()} for p, ds in groups.items()}


def _per_seed(records: Sequence[RunRecord], label: str, threshold: float) -> Stat:
    by_seed: dict[int, list[RunRecord]] = {}
    for r in records:
        if r.label == label:
            by_seed.setdefault(r.seed, []).append(r)
    return mean_ci(_rate(rs, threshold) for _, rs in sorted(by_seed.items()))


def _mean(values: Iterable[float | None]) -> float:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.fmean(xs) if xs else math.nan


def _median(values: Iterable[int | None]) -> float:
    xs = [v for v in values if v is not None]
    return float(statistics.median(xs)) if xs else math.nan


@dataclass(slots=True)
class Summary:
    """Tables behind summary.csv (per detector) and the profile x detector breakdown."""

    detectors: list[dict[str, Any]]
    profiles: list[dict[str, Any]]
    matrix: dict[str, dict[str, float]]


def compute_summary(records: Sequence[RunRecord], threshold: float = 0.5) -> Summary:
    """Per detector: DR and FPR over seeds (mean ± std, 95 % CI), deltas, time-to-detect, WTSD.

    ``delta_dr_vs_baseline`` = DR(bot profiles) - flag rate of their auto-baseline twins (paired by
    profile); ``delta_fpr_vs_ref`` = FPR(detector) - FPR(first detector in the config).
    """
    det_ids = list(dict.fromkeys(r.detector_id for r in records))
    matrix = detection_matrix(records, threshold)
    labels = {r.profile_id: (r.label, r.twin_of) for r in records}
    twins = {twin_of: p for p, (label, twin_of) in labels.items() if twin_of is not None}

    profile_rows: list[dict[str, Any]] = []
    for p, (label, twin_of) in labels.items():
        for d in det_ids:
            rs = [r for r in records if r.profile_id == p and r.detector_id == d]
            ok = [r for r in rs if r.ok]
            flags = mean_ci(float(r.flagged(threshold)) for r in ok)
            twin = twins.get(p)
            delta = matrix[p][d] - matrix[twin][d] if twin is not None and d in matrix.get(twin, {}) else math.nan
            profile_rows.append({
                "profile_id": p, "label": label, "twin_of": twin_of or "", "detector_id": d,
                "runs_ok": len(ok), "runs_failed": len(rs) - len(ok),
                "detection_rate": flags.mean, "dr_std": flags.std, "dr_ci_lo": flags.lo, "dr_ci_hi": flags.hi,
                "mean_bot_belief": _mean(r.bot_belief() for r in ok), "wtsd_mean": _mean(r.wtsd for r in ok),
                "ttd_median": _median(time_to_detect(r.prefix, threshold) for r in ok),
                "delta_dr_vs_baseline": delta,
            })

    det_rows: list[dict[str, Any]] = []
    ref_fpr = math.nan
    for i, d in enumerate(det_ids):
        rs = [r for r in records if r.detector_id == d]
        dr, fpr = _per_seed(rs, "bot", threshold), _per_seed(rs, "baseline", threshold)
        if i == 0:
            ref_fpr = fpr.mean
        deltas = [row["delta_dr_vs_baseline"] for row in profile_rows if row["detector_id"] == d]
        bots = [r for r in rs if r.ok and r.label == "bot"]
        det_rows.append({
            "detector_id": d, "detector_version": rs[0].detector_version if rs else "",
            "runs_ok": sum(r.ok for r in rs), "runs_failed": sum(not r.ok for r in rs),
            "detection_rate": dr.mean, "dr_std": dr.std, "dr_ci_lo": dr.lo, "dr_ci_hi": dr.hi,
            "fpr": fpr.mean, "fpr_std": fpr.std, "fpr_ci_lo": fpr.lo, "fpr_ci_hi": fpr.hi, "n_seeds": dr.n,
            "delta_dr_vs_baseline": _mean(deltas), "delta_fpr_vs_ref": fpr.mean - ref_fpr,
            "ttd_median": _median(time_to_detect(r.prefix, threshold) for r in bots),
            "wtsd_bot_mean": _mean(r.wtsd for r in bots),
            "wtsd_baseline_mean": _mean(r.wtsd for r in rs if r.ok and r.label == "baseline"),
        })
    return Summary(det_rows, profile_rows, matrix)


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 6)
    return v


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], headers: Sequence[str] | None = None) -> Path:
    """Rows of dicts to CSV (NaN as empty, floats rounded to 6 digits)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cols = list(headers or (rows[0].keys() if rows else []))
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([_cell(row.get(c, "")) for c in cols])
    return target
