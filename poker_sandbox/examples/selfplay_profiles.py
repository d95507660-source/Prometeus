"""Three heuristic profiles at one 6-max table (2 seats each), stacks reset to 100 bb every hand.

Each profile plays every hand on two seats, so 10k hands = 20k seat-hands per profile.
Deterministic for a fixed --seed.

    python examples/selfplay_profiles.py --hands 10000 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

from adapter.config import AdapterConfig, Stakes  # noqa: E402
from adapter.mock import MockAdapter  # noqa: E402
from strategy import AGGRESSIVE, LOOSE, TIGHT, HeuristicStrategy, collect, play_hands  # noqa: E402

PROFILES = {"tight": TIGHT, "loose": LOOSE, "aggressive": AGGRESSIVE}


def main() -> None:
    """Run the session and print a stats table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    stakes = Stakes(small_blind=1, big_blind=2, starting_stack=200)
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=6, seed=args.seed, auto_deal=False, hero_seat=None, stakes=stakes))
    names = list(PROFILES)
    groups = {seat: names[seat % len(names)] for seat in range(6)}
    strategies = {
        seat: HeuristicStrategy(replace(PROFILES[g], seed=args.seed * 100 + seat), name=g)
        for seat, g in groups.items()
    }
    started = time.perf_counter()
    results = play_hands(mock, strategies, args.hands, stakes.starting_stack)
    elapsed = time.perf_counter() - started

    digest = hashlib.sha256()
    for r in results:
        for rec in r.actions:
            digest.update(f"{rec.seat}:{rec.type.value}:{rec.amount}|".encode())

    print(f"hands={len(results)} seed={args.seed} time={elapsed:.1f}s actions_checksum={digest.hexdigest()[:16]}")
    print(f"{'profile':11} {'seat-hands':>10} {'VPIP':>6} {'PFR':>6} {'3bet':>6} {'AF':>5} {'WTSD':>6} "
          f"{'bb/100':>8} {'±1.96SE':>8}")
    for name, stats in collect(results, groups, stakes.big_blind).items():
        q = stats.ratios()
        print(f"{name:11} {int(q['hands']):>10} {q['vpip']:6.1f} {q['pfr']:6.1f} {q['3bet']:6.1f} {q['af']:5.2f} "
              f"{q['wtsd']:6.1f} {q['bb100']:8.1f} {1.96 * q['bb100_se']:8.1f}")
    mock.disconnect()


if __name__ == "__main__":
    main()
