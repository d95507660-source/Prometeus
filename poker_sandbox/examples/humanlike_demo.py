"""6-max self-play at three humanlike levels (0, 0.5, 1) over the same deck seed.

Synthetic bots only, offline mock, deterministic for a fixed --seed (log timestamps aside).
Every level shares the deck seed and bot seeds: levels are compared to each other, not to
a baseline on another seed.

    python examples/humanlike_demo.py --hands 1000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

from adapter.config import AdapterConfig  # noqa: E402
from adapter.mock import MockAdapter  # noqa: E402
from adapter.models import Action, TableState  # noqa: E402
from behavior import BehaviorLogger, HumanLikeProfile, HumanLikeWrapper, timing_report  # noqa: E402
from collusion.runner import legality_error  # noqa: E402
from strategy import HeuristicParams, HeuristicStrategy, collect, play_hands  # noqa: E402

SEATS = 6
STACK = 200
LEVELS = (0.0, 0.5, 1.0)


def run_level(level: float, hands: int, seed: int, out_dir: Path) -> tuple[list[str], Path]:
    """Play ``hands`` hands with every seat wrapped at ``level``; return a table row and the log path."""
    path = out_dir / f"behavior_level_{level:.2f}.jsonl"
    path.unlink(missing_ok=True)  # the log appends; start each run clean
    profile = replace(HumanLikeProfile(session_hands=hands), humanlike_level=level)
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=SEATS, seed=seed, auto_deal=False, hero_seat=None, table_id="humanlike"))

    def check(seat: int, state: TableState, action: Action) -> None:
        if (error := legality_error(seat, state, action)) is not None:
            raise AssertionError(f"illegal decision at {state.hand_id}: {error}")

    with BehaviorLogger(path) as log:
        wrappers = {
            seat: HumanLikeWrapper(HeuristicStrategy(HeuristicParams(seed=seed * 1000 + seat)), profile,
                                   seed=seed * 7919 + seat, seat=seat, behavior_log=log)
            for seat in range(SEATS)
        }
        results = play_hands(mock, wrappers, hands, STACK, check)
    mock.disconnect()
    records = [r for w in wrappers.values() for r in w.records]
    t = timing_report([r.planned_delay_ms for r in records])
    errors = sum(r.error_applied for r in records) / len(records)
    tilt = sum(len(w.tilt_hand_ids) for w in wrappers.values()) / sum(w.hands_seen for w in wrappers.values())
    stats = collect(results, {s: "all" for s in range(SEATS)}, 2)["all"].ratios()
    row = [f"{level:5.2f}", f"{t.n:6d}", f"{t.mean:8.0f}", f"{t.p50:8.0f}", f"{t.p95:8.0f}", f"{t.p99:8.0f}",
           f"{t.cv:6.3f}", f"{100 * errors:6.2f}", f"{100 * tilt:6.2f}", f"{stats['wtsd']:6.1f}"]
    return row, path


def main() -> None:
    """Run all levels and print the table and log paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "humanlike")
    args = parser.parse_args()

    header = ["level", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "CV", "err%", "tilt%", "WTSD%"]
    widths = [5, 6, 8, 8, 8, 8, 6, 6, 6, 6]
    started = time.perf_counter()
    rows, paths = [], []
    for level in LEVELS:
        row, path = run_level(level, args.hands, args.seed, args.out)
        rows.append(row)
        paths.append(path)
    print(f"hands={args.hands} seats={SEATS} seed={args.seed} time={time.perf_counter() - started:.1f}s")
    print(" ".join(h.rjust(w) for h, w in zip(header, widths, strict=True)))
    for row in rows:
        print(" ".join(row))
    print("bluff_bonus: HeuristicStrategy has no `bluff_bonus` attribute -> tilt bluffing ignored")
    for path in paths:
        print(f"behavior log: {path}")


if __name__ == "__main__":
    main()
