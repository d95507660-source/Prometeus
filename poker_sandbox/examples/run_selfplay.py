"""Self-play: RandomStrategy on every seat of a MockAdapter, heads-up and 6-max.

Deterministic for a fixed --seed: actions, hand histories, final stacks and the
sequence of state hashes repeat exactly (only ts/latency_ms differ).

    python examples/run_selfplay.py --hands 1000 --seed 7
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

from adapter.config import AdapterConfig, Stakes  # noqa: E402
from adapter.mock import MockAdapter  # noqa: E402
from adapter.models import Event, EventType  # noqa: E402
from agent import Agent, AgentConfig, AgentMetrics, DecisionLogger, RandomStrategy, SeatBoundAdapter  # noqa: E402

HAND_TIMEOUT_S = 30.0


async def play(seats: int, hands: int, seed: int, out_dir: Path) -> dict[str, object]:
    """Play ``hands`` hands at a ``seats``-max table and return a report."""
    decision_log = out_dir / f"selfplay_{seats}max_decisions.jsonl"
    audit_log = out_dir / f"selfplay_{seats}max_audit.jsonl"
    for path in (decision_log, audit_log):
        path.unlink(missing_ok=True)  # both logs append; start clean for reproducible files

    stakes = Stakes()
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=seats, seed=seed, auto_deal=False, hero_seat=None,
                               stakes=stakes, log_path=audit_log))
    bound = SeatBoundAdapter.for_seats(mock, range(seats))
    driver = bound[0]  # any wrapper: they share the lock

    loop = asyncio.get_running_loop()
    hand_done = asyncio.Event()

    def on_driver_event(event: Event) -> None:
        if event.type is EventType.HAND_ENDED:
            loop.call_soon_threadsafe(hand_done.set)

    # Driver subscribes as an observer (hero_seat=None → public events only).
    unsubscribe = driver.run_locked(lambda: mock.on_event(on_driver_event))

    def deal() -> None:
        for s in range(seats):
            mock.set_stack(s, stakes.starting_stack)
        mock.start_hand()

    with DecisionLogger(decision_log) as logger:
        agents = [Agent(bound[s], RandomStrategy(seed + s), AgentConfig(), decision_logger=logger)
                  for s in range(seats)]
        tasks = [asyncio.create_task(a.run()) for a in agents]
        try:
            await asyncio.gather(*(a.wait_started() for a in agents))
            for _ in range(hands):
                hand_done.clear()
                await asyncio.to_thread(driver.run_locked, deal)
                await asyncio.wait_for(hand_done.wait(), HAND_TIMEOUT_S)
        finally:
            await asyncio.gather(*(a.stop() for a in agents))
            await asyncio.gather(*tasks)
            driver.run_locked(unsubscribe)

    hand_ids = mock.hand_ids
    results = [driver.get_hand_history(h) for h in hand_ids]
    driver.disconnect()

    agent_seats = {a.seat for a in agents}
    records = [r for res in results for r in res.actions if r.seat in agent_seats]
    total = AgentMetrics()
    for a in agents:
        total = total.merge(a.metrics)
    if total.decisions_total != len(records):
        raise AssertionError(f"metrics {total.decisions_total} != history {len(records)}")

    digest = hashlib.sha256()
    for res in results:
        digest.update(res.hand_id.encode())
        for r in res.actions:
            digest.update(f"{r.seat}:{r.type.value}:{r.amount}|".encode())
        digest.update(json.dumps(sorted(res.final_stacks.items())).encode())
    hashes = hashlib.sha256()
    for line in decision_log.read_text().splitlines():
        hashes.update(json.loads(line)["state_hash"].encode())

    return {
        "seats": seats,
        "hands": len(hand_ids),
        "metrics": total.snapshot(),
        "history_action_records": len(records),
        "actions_checksum": digest.hexdigest()[:16],
        "state_hashes_checksum": hashes.hexdigest()[:16],
        "decision_log": str(decision_log),
        "audit_log": str(audit_log),
    }


async def main_async(args: argparse.Namespace) -> None:
    """Run every requested table size sequentially and print reports."""
    for seats in args.seats:
        report = await play(seats, args.hands, args.seed, args.out)
        print(json.dumps(report, indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seats", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--out", type=Path, default=ROOT / "logs")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
