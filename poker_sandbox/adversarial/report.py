"""Analysis of search results: Spearman per field, top problematic configs, field x bin matrix, CSV + Markdown."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adversarial.search import Evaluation, Mode
from adversarial.space import ChoiceAxis, FloatAxis, IntAxis, ParamSpace

__all__ = [
    "BinCell",
    "ReportPaths",
    "bin_matrix",
    "correlations",
    "markdown_table",
    "matrix_rows",
    "ranked",
    "spearman",
    "top_problematic",
    "top_rows",
    "unique",
    "write_report",
    "wtsd_driven",
]


def unique(evals: Sequence[Evaluation]) -> list[Evaluation]:
    """First evaluation per config hash (cache hits repeat configs across phases)."""
    seen: set[str] = set()
    out = []
    for e in evals:
        if e.config_hash not in seen:
            seen.add(e.config_hash)
            out.append(e)
    return out


def _ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = arr.argsort(kind="stable")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    for v in np.unique(arr):  # average ranks for ties
        mask = arr == v
        ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation; NaN for fewer than 3 points or a constant side."""
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    if len(xs) < 3:
        return math.nan
    rx, ry = _ranks(xs), _ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _value(e: Evaluation, name: str) -> float:
    return float(getattr(e.config, name))  # bools → 0/1


def correlations(evals: Sequence[Evaluation], space: ParamSpace, mode: Mode) -> dict[str, float]:
    """Spearman of every numeric field with fitness (unique configs)."""
    evs = unique(evals)
    fit = [e.fitness(mode) for e in evs]
    return {name: spearman([_value(e, name) for e in evs], fit) for name in space.numeric_fields}


def top_problematic(evals: Sequence[Evaluation], mode: Mode, k: int = 10) -> list[Evaluation]:
    """``detect``: bots the detector misses most confidently; ``evade``: configs caught most easily."""
    sign = 1.0 if mode == "detect" else -1.0
    return sorted(unique(evals), key=lambda e: sign * e.score.bot_belief)[:k]


@dataclass(frozen=True, slots=True)
class BinCell:
    """Mean fitness of both directions and caught share over the configs in one bin of a field."""

    label: str
    n: int
    detect: float
    evade: float
    caught: float


def _bins(space: ParamSpace, name: str, n_bins: int) -> list[tuple[str, float, float]]:
    axis = space.axis(name)
    if isinstance(axis, ChoiceAxis):
        return [(str(c), float(c), float(c)) for c in axis.choices]
    if isinstance(axis, IntAxis) and axis.hi - axis.lo + 1 <= n_bins:
        return [(str(v), float(v), float(v)) for v in range(axis.lo, axis.hi + 1)]
    assert isinstance(axis, FloatAxis | IntAxis)
    edges = np.linspace(axis.lo, axis.hi, n_bins + 1)
    return [(f"{edges[i]:.3g}..{edges[i + 1]:.3g}", float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]


def bin_matrix(evals: Sequence[Evaluation], space: ParamSpace, n_bins: int = 3) -> dict[str, list[BinCell]]:
    """Field x bin: mean detect fitness (= bot belief), mean evade fitness and share flagged as bot."""
    evs = unique(evals)
    out: dict[str, list[BinCell]] = {}
    for name in space.numeric_fields:
        bins = _bins(space, name, n_bins)
        cells = []
        for i, (label, lo, hi) in enumerate(bins):
            last = i == len(bins) - 1
            members = [e for e in evs if lo <= _value(e, name) <= hi and (last or lo == hi or _value(e, name) < hi)]
            n = len(members)
            cells.append(BinCell(label, n,
                                 sum(e.fitness("detect") for e in members) / n if n else math.nan,
                                 sum(e.fitness("evade") for e in members) / n if n else math.nan,
                                 sum(e.score.is_bot for e in members) / n if n else math.nan))
        out[name] = cells
    return out


def wtsd_driven(evals: Sequence[Evaluation]) -> list[Evaluation]:
    """Flagged configs whose largest component score is WTSD (needs ``s_*`` features, as FakeOracle gives)."""
    out = []
    for e in unique(evals):
        parts = {k: v for k, v in e.score.features.items() if k.startswith("s_") and not math.isnan(v)}
        if e.score.is_bot and parts and max(parts, key=lambda k: parts[k]) == "s_wtsd":
            out.append(e)
    return out


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


TOP_FIELDS = ("vpip_scale", "aggression", "bluff_freq", "humanlike_level", "latency_sigma", "collusion",
              "group_size")
TOP_FEATURES = ("timing_cv", "vpip", "wtsd", "yield_rate")


def top_rows(evals: Sequence[Evaluation], mode: Mode) -> tuple[list[str], list[list[object]]]:
    """Headers and rows of the top-problematic table."""
    headers = ["#", "hash", "phase", *TOP_FIELDS, "is_bot", "conf", *TOP_FEATURES, "reason"]
    rows: list[list[object]] = []
    for i, e in enumerate(top_problematic(evals, mode), 1):
        rows.append([i, e.config_hash[:8], e.phase, *(getattr(e.config, f) for f in TOP_FIELDS), e.score.is_bot,
                     e.score.confidence, *(float(e.score.features.get(f, math.nan)) for f in TOP_FEATURES),
                     e.score.reason])
    return headers, rows


def matrix_rows(evals: Sequence[Evaluation], space: ParamSpace) -> tuple[list[str], list[list[object]]]:
    """Headers and rows of the field x bin matrix (cell = "detect fitness / caught share (n)")."""
    matrix = bin_matrix(evals, space)
    width = max(len(cells) for cells in matrix.values())
    headers = ["field", *(f"bin{i + 1}" for i in range(width))]
    rows: list[list[object]] = []
    for name, cells in matrix.items():
        row: list[object] = [name]
        for c in cells:
            row.append(f"{c.label}: {_fmt(c.detect)} / {_fmt(c.caught)} (n={c.n})")
        rows.append(row + [""] * (width - len(cells)))
    return headers, rows


def ranked(corr: dict[str, float]) -> list[tuple[str, float]]:
    """Strongest |rho| first, NaN last."""
    return sorted(corr.items(), key=lambda kv: 1.0 if math.isnan(kv[1]) else -abs(kv[1]))


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Files written by :func:`write_report`."""

    csv: Path
    markdown: Path


def write_report(evals: Sequence[Evaluation], space: ParamSpace, mode: Mode, out_dir: Path | str) -> ReportPaths:
    """All unique evaluations to CSV; correlations, top-10, matrix and WTSD-driven configs to Markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evs = unique(evals)
    feature_names = sorted({k for e in evs for k in e.score.features})
    field_names = [a.name for a in space.axes]
    csv_path = out / "adversarial_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_hash", "phase", "generation", *field_names, "is_bot", "confidence", "bot_belief",
                         f"fitness_{mode}", *feature_names, "reason"])
        for e in evs:
            writer.writerow([e.config_hash, e.phase, "" if e.generation is None else e.generation,
                             *(getattr(e.config, f) for f in field_names), e.score.is_bot, e.score.confidence,
                             e.score.bot_belief, e.fitness(mode),
                             *(_fmt(float(e.score.features.get(f, math.nan))) for f in feature_names),
                             e.score.reason])

    corr = correlations(evs, space, mode)
    th, tr = top_rows(evs, mode)
    mh, mr = matrix_rows(evs, space)
    wtsd = wtsd_driven(evs)
    caught = sum(e.score.is_bot for e in evs)
    md = [
        f"# Adversarial search report (mode: {mode})",
        "",
        f"Unique configs: {len(evs)}; flagged as bot: {caught} ({100 * caught / max(len(evs), 1):.1f} %).",
        "",
        f"## Spearman(field, fitness_{mode})",
        "",
        markdown_table(["field", "spearman"], [[k, v] for k, v in ranked(corr)]),
        "",
        "## Top-10 problematic configs " + ("(missed bots)" if mode == "detect" else "(caught most easily)"),
        "",
        markdown_table(th, tr),
        "",
        "## Field x bin: mean detect fitness / caught share (n)",
        "",
        markdown_table(mh, mr),
        "",
        f"## Flagged mainly through WTSD: {len(wtsd)} configs",
        "",
        markdown_table(["hash", "wtsd", "timing_cv", "confidence"],
                       [[e.config_hash[:8], float(e.score.features.get("wtsd", math.nan)),
                         float(e.score.features.get("timing_cv", math.nan)), e.score.confidence] for e in wtsd]),
        "",
    ]
    md_path = out / "adversarial_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return ReportPaths(csv_path, md_path)
