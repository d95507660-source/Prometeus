"""6-max, a 2-seat coordinating group vs solo bots, plus the same table without coordination.

Synthetic bots only, offline mock, deterministic for a fixed --seed (log timestamps aside).

    python examples/collusion_demo.py --hands 1000 --seed 1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

from collusion import CoordinationParams, ScenarioConfig, run_experiment  # noqa: E402

ROWS = (
    ("sync_rate", "sync (pressure+fold pairs)"),
    ("yield_rate", "yield (fold to partner aggression)"),
    ("sizing_pearson", "sizing corr, Pearson"),
    ("sizing_spearman", "sizing corr, Spearman"),
    ("group_bb100", "group bb/100"),
    ("group_top1_share", "group top-1 share per hand"),
)


def _fmt(value: float) -> str:
    return "   n/a" if math.isnan(value) else f"{value:8.3f}"


def main() -> None:
    """Run the experiment and print metrics and log paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "collusion")
    args = parser.parse_args()

    cfg = ScenarioConfig(n_seats=6, group_seats=(0, 1), coordination=CoordinationParams(),
                         n_hands=args.hands, seed=args.seed, out_dir=args.out)
    started = time.perf_counter()
    exp = run_experiment(cfg)
    m = exp.metrics
    print(f"hands={args.hands} seed={args.seed} group={cfg.group_seats} time={time.perf_counter() - started:.1f}s")
    print(f"signals sent={exp.coordinated.signals_sent} adjustments={exp.coordinated.adjustments}")
    print(f"{'metric':36} {'group':>8} {'baseline':>8}")
    for key, label in ROWS:
        print(f"{label:36} {_fmt(m[key + '_group'])} {_fmt(m[key + '_baseline'])}")
    print(f"{'pairs: sync / yield / sizing':36} "
          f"{m['sync_pairs_group']:.0f}/{m['yield_opps_group']:.0f}/{m['sizing_pairs_group']:.0f}  "
          f"{m['sync_pairs_baseline']:.0f}/{m['yield_opps_baseline']:.0f}/{m['sizing_pairs_baseline']:.0f}")
    print(f"profit paired z={_fmt(m['profit_paired_z'])}  top-1 share z={_fmt(m['top1_share_z'])}")
    print(f"signal log:     {exp.coordinated.signal_log}")
    print(f"adjustment log: {exp.coordinated.adjustment_log}")
    print(f"metrics csv:    {exp.metrics_csv}")


if __name__ == "__main__":
    main()
