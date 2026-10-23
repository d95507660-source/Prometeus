"""Reports of an experiment: Markdown, summary/matrix/runs_flat CSV and manifest.json."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adversarial.report import markdown_table
from harness.config import DetectorSpec, ExperimentConfig
from harness.env import BATCH_SIZE, versions
from harness.metrics import Summary, compute_summary, time_to_detect, write_csv
from harness.runner import RunRecord, jsonable

__all__ = ["RECOMMENDED_SEEDS", "ReportPaths", "build_manifest", "runs_flat", "write_reports"]

RECOMMENDED_SEEDS = 20
CONFIG_COLUMNS = ("vpip_scale", "pfr_scale", "aggression", "bluff_freq", "humanlike_level", "collusion",
                  "group_size", "coordination_strength", "signal_interval")
SUMMARY_COLUMNS = ("detector_id", "detector_version", "runs_ok", "runs_failed", "n_seeds", "detection_rate",
                   "dr_std", "dr_ci_lo", "dr_ci_hi", "fpr", "fpr_std", "fpr_ci_lo", "fpr_ci_hi",
                   "delta_dr_vs_baseline", "delta_fpr_vs_ref", "ttd_median", "wtsd_bot_mean", "wtsd_baseline_mean")


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Files written by :func:`write_reports`."""

    markdown: Path
    summary_csv: Path
    matrix_csv: Path
    runs_flat_csv: Path
    manifest: Path
    summary: Summary


def runs_flat(records: Sequence[RunRecord], threshold: float) -> list[dict[str, Any]]:
    """One flat row per run (no volatile timing fields, so the CSV is reproducible)."""
    metric_names = list(dict.fromkeys(k for r in records for k in r.metrics))
    rows = []
    for r in records:
        score = r.score or {}
        row: dict[str, Any] = {
            "profile_id": r.profile_id, "label": r.label, "detector_id": r.detector_id, "seed": r.seed,
            "status": r.status, "is_bot": score.get("is_bot", ""), "confidence": score.get("confidence", math.nan),
            "bot_belief": r.bot_belief(), "flagged": r.flagged(threshold),
            "ttd": time_to_detect(r.prefix, threshold) if r.ok else "", "wtsd": _num(r.wtsd),
            "reason": score.get("reason", ""), "config_hash": r.config_hash, "cache_key": r.cache_key,
        }
        row.update({f"m_{k}": _num(r.metrics.get(k)) for k in metric_names})
        row.update({f"c_{k}": r.config.get(k, "") for k in CONFIG_COLUMNS})
        row["error"] = r.error or ""
        rows.append(row)
    return rows


def _num(v: Any) -> float:
    return math.nan if v is None else float(v)


def build_manifest(exp: ExperimentConfig, specs: Sequence[DetectorSpec], records: Sequence[RunRecord], *,
                   git: str, started: str, ended: str, files: Sequence[Path]) -> dict[str, Any]:
    """Everything needed to reproduce (and audit) the run."""
    manifest: dict[str, Any] = jsonable({
        "experiment": exp.name,
        "config_hash": exp.config_hash(),
        "config": exp.model_dump(mode="json"),
        "versions": versions(),
        "git_hash": git or "unknown",
        "seeds": list(exp.seeds),
        "n_hands_per_run": exp.n_hands_per_run,
        "batch_size": BATCH_SIZE,
        "equity_samples": exp.equity_samples,
        "threshold": exp.threshold,
        "started_at": started,
        "ended_at": ended,
        "detectors": [{"name": d.name, "kind": d.kind, "version": d.version, "params": d.params,
                       "detector_id": d.detector_id, "stub": d.kind == "fake"} for d in specs],
        "runs": {"total": len(records), "ok": sum(r.ok for r in records), "failed": sum(not r.ok for r in records),
                 "cached": sum(r.cached for r in records)},
        "files": [p.name for p in files],
    })
    return manifest


def _run_rows(records: Sequence[RunRecord], threshold: float) -> list[list[Any]]:
    out: list[list[Any]] = []
    for r in records:
        score = r.score or {}
        out.append([r.profile_id, r.detector_id.split("#")[0], r.seed, r.bot_belief(), _num(r.wtsd),
                     time_to_detect(r.prefix, threshold) or "-", *(r.config.get(k) for k in CONFIG_COLUMNS[:7]),
                     score.get("reason", "")])
    return out


def write_reports(exp: ExperimentConfig, records: Sequence[RunRecord], out_dir: Path | str, *,
                  specs: Sequence[DetectorSpec], git: str, started: str, ended: str) -> ReportPaths:
    """Write report.md, summary.csv, matrix.csv, runs_flat.csv and manifest.json into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    th = exp.threshold
    summary = compute_summary(records, th)
    det_ids = [d.detector_id for d in specs]

    summary_csv = write_csv(out / "summary.csv", summary.detectors, SUMMARY_COLUMNS)
    matrix_rows = [{"profile_id": p, **{d: row.get(d, math.nan) for d in det_ids}} for p, row in summary.matrix.items()]
    matrix_csv = write_csv(out / "matrix.csv", matrix_rows, ["profile_id", *det_ids])
    flat_csv = write_csv(out / "runs_flat.csv", runs_flat(records, th))

    bots = [r for r in records if r.ok and r.label == "bot"]
    fn = sorted((r for r in bots if not r.flagged(th)), key=lambda r: (r.bot_belief(), r.profile_id, r.seed))[:5]
    tp = sorted((r for r in bots if r.flagged(th)), key=lambda r: (-r.bot_belief(), r.profile_id, r.seed))[:5]
    run_headers = ["profile", "detector", "seed", "bot_belief", "WTSD", "ttd", *CONFIG_COLUMNS[:7], "reason"]
    failed = [r for r in records if not r.ok]

    md = [f"# Experiment `{exp.name}`", "",
          f"- config hash: `{exp.config_hash()}`; git: `{git or 'unknown'}`; started {started}, ended {ended}",
          f"- {len(records)} runs ({sum(r.ok for r in records)} ok, {len(failed)} failed, "
          f"{sum(r.cached for r in records)} from cache); {exp.n_hands_per_run} hands/run; "
          f"seeds {list(exp.seeds)}; threshold {th}; equity_samples {exp.equity_samples}; BATCH_SIZE {BATCH_SIZE}",
          "- detectors: " + "; ".join(f"`{d.name}` kind={d.kind} version={d.version}"
                                      + (" (**FakeOracle stub**, not a real detector)" if d.kind == "fake" else "")
                                      for d in specs)]
    if len(exp.seeds) < RECOMMENDED_SEEDS:
        md.append(f"- **{len(exp.seeds)} seeds only: use >= {RECOMMENDED_SEEDS} seeds for real comparisons** "
                  "(CIs below are wide).")
    if exp.notes:
        md.append(f"- notes: {exp.notes}")
    md += ["", "Baseline twins (`<profile>__baseline`) keep the strategy knobs without coordination and "
           "human-like noise; they form the negative class for FPR. All players are synthetic bots, so "
           "\"FPR\" is the flag rate on that reference, not on humans.", "",
           "## Summary per detector (mean ± std over seeds, 95 % CI)", "",
           markdown_table(["detector", "version", "ok/failed", "detection rate", "FPR", "Δ DR vs baseline",
                           "Δ FPR vs ref", "ttd median", "WTSD bots", "WTSD baseline"],
                          [[r["detector_id"], r["detector_version"], f"{r['runs_ok']}/{r['runs_failed']}",
                            _stat(r, "detection_rate", "dr"), _stat(r, "fpr", "fpr"), r["delta_dr_vs_baseline"],
                            r["delta_fpr_vs_ref"], r["ttd_median"], r["wtsd_bot_mean"], r["wtsd_baseline_mean"]]
                           for r in summary.detectors]),
           "", "## Detection rate: profile x detector", "",
           markdown_table(["profile", *(d.split("#")[0] for d in det_ids)],
                          [[p, *(row.get(d, math.nan) for d in det_ids)] for p, row in summary.matrix.items()]),
           "", "## Per profile", "",
           markdown_table(["profile", "label", "detector", "DR (mean ± std)", "mean belief", "WTSD", "ttd median",
                           "Δ DR vs twin"],
                          [[r["profile_id"], r["label"], r["detector_id"].split("#")[0],
                            f"{r['detection_rate']:.3f} ± {r['dr_std']:.3f}", r["mean_bot_belief"], r["wtsd_mean"],
                            r["ttd_median"], r["delta_dr_vs_baseline"]] for r in summary.profiles]),
           "", "## Top-5 worst runs (false negatives: bots not flagged)", "",
           markdown_table(run_headers, _run_rows(fn, th)) if fn else "_none_",
           "", "## Top-5 best runs (true positives, highest confidence)", "",
           markdown_table(run_headers, _run_rows(tp, th)) if tp else "_none_", ""]
    if failed:
        md += ["## Failed runs", "", markdown_table(["profile", "detector", "seed", "error"],
                                                    [[r.profile_id, r.detector_id, r.seed, r.error] for r in failed]),
               ""]
    md += ["Files: summary.csv, matrix.csv, runs_flat.csv, runs.jsonl, manifest.json", ""]
    md_path = out / "report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    manifest_path = out / "manifest.json"
    files = [md_path, summary_csv, matrix_csv, flat_csv, out / "runs.jsonl"]
    manifest = build_manifest(exp, specs, records, git=git, started=started, ended=ended, files=files)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ReportPaths(md_path, summary_csv, matrix_csv, flat_csv, manifest_path, summary)


def _stat(row: dict[str, Any], mean_key: str, prefix: str) -> str:
    mean = row[mean_key]
    if isinstance(mean, float) and math.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {row[prefix + '_std']:.3f} [{row[prefix + '_ci_lo']:.3f}, {row[prefix + '_ci_hi']:.3f}]"
