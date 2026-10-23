"""Command line: ``python -m harness run --config examples/exp_demo.json [--workers N] ...``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from harness.config import ExperimentConfig
from harness.runner import run_experiment

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI."""
    parser = argparse.ArgumentParser(prog="python -m harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("--config", type=Path, required=True, help="experiment JSON")
    run.add_argument("--workers", type=int, default=1, help="parallel tables (processes)")
    run.add_argument("--detector", action="append", default=[], help="keep only this detector name/kind (repeatable)")
    run.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    run.add_argument("--resume", action="store_true", help="reuse cached ok runs; skip known failures")
    run.add_argument("--retry-failed", action="store_true", help="with --resume: rerun known failures")
    run.add_argument("--output-dir", type=Path, default=None, help="override output_dir")
    run.add_argument("--cache-dir", type=Path, default=None, help="cache directory (default logs/harness/cache)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code (1 if any run failed)."""
    args = build_parser().parse_args(argv)
    exp = ExperimentConfig.from_file(args.config)
    started = time.perf_counter()
    result = run_experiment(exp, workers=args.workers, resume=args.resume, retry_failed=args.retry_failed,
                            detectors=args.detector, dry_run=args.dry_run, output_dir=args.output_dir,
                            cache_dir=args.cache_dir)
    if args.dry_run:
        print("[harness] dry run: nothing executed")
        return 0
    assert result.report is not None
    elapsed = time.perf_counter() - started
    print(f"[harness] done in {elapsed:.1f}s: tables run {result.tables_run}, cached runs {result.cached}, "
          f"failed runs {result.failed}, skipped known failures {result.skipped_failed}")
    print(f"{'detector':40} {'detection rate (mean ± std [95% CI])':42} {'FPR (mean ± std [95% CI])':42} "
          f"{'ttd':>5} {'WTSD bot':>8}")
    for row in result.report.summary.detectors:
        dr = f"{row['detection_rate']:.3f} ± {row['dr_std']:.3f} [{row['dr_ci_lo']:.3f}, {row['dr_ci_hi']:.3f}]"
        fpr = f"{row['fpr']:.3f} ± {row['fpr_std']:.3f} [{row['fpr_ci_lo']:.3f}, {row['fpr_ci_hi']:.3f}]"
        print(f"{row['detector_id']:40} {dr:42} {fpr:42} {row['ttd_median']:5.0f} {row['wtsd_bot_mean']:8.1f}")
    print(f"report:   {result.report.markdown}")
    print(f"manifest: {result.report.manifest}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
