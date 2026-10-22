"""Equity throughput benchmark (CPU, single process).

    python examples/equity_bench.py --sims 500000
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

import numpy as np  # noqa: E402

from pokermath import BATCH_SIZE, Range, equity_vs_range_with_ci, equity_with_ci, parse_many  # noqa: E402

TARGET_HU_PER_S = 100_000


def bench(fn: Callable[[int, int], tuple[float, float]], sims: int) -> tuple[float, float, float, float]:
    """Return (warm-up seconds, sims/s, equity, standard error); warm-up builds caches and pages memory."""
    t0 = time.perf_counter()
    fn(BATCH_SIZE, 0)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    eq, se = fn(sims, 1)
    elapsed = time.perf_counter() - t0
    return warm, sims / elapsed, eq, se


def main() -> None:
    """Run the three cases and print a table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500_000)
    args = parser.parse_args()

    river_board = parse_many("Ah 7d 2c Ts 3h")
    river_range = Range.parse("22+, ATs+, KQs, ATo+, KJo+, T9s, 98s")
    cases: list[tuple[str, Callable[[int, int], tuple[float, float]]]] = [
        ("HU preflop: AA vs random", lambda n, s: equity_with_ci(parse_many("As Ah"), (), 1, n, s)),
        ("3-way preflop: AKs vs 2 random", lambda n, s: equity_with_ci(parse_many("As Ks"), (), 2, n, s)),
        ("River all-in: AhKd vs range", lambda n, s: equity_vs_range_with_ci(
            parse_many("Ad Kd"), river_board, river_range, n, s)),
    ]
    print(f"python {platform.python_version()}, numpy {np.__version__}, {platform.machine()}, sims={args.sims}")
    print(f"{'case':34} {'warm-up s':>10} {'sims/s':>12} {'equity':>8} {'±1.96SE':>8}")
    rates = []
    for name, fn in cases:
        warm, rate, eq, se = bench(fn, args.sims)
        rates.append(rate)
        print(f"{name:34} {warm:10.3f} {rate:12,.0f} {eq:8.4f} {1.96 * se:8.4f}")
    verdict = "OK" if rates[0] >= TARGET_HU_PER_S else "BELOW TARGET"
    print(f"HU preflop target >= {TARGET_HU_PER_S:,} sims/s: {verdict}")


if __name__ == "__main__":
    main()
