"""Scenario runner: a coordinated group vs solo bots, and the same table without coordination.

Synthetic only: every seat is our own bot on the offline M0 mock, driven by
``strategy.sim.play_hands``. Both runs share the deck seed, so hands are paired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from adapter.config import AdapterConfig, Stakes
from adapter.mock import MockAdapter
from adapter.models import Action, ActionType, HandResult, TableState
from agent.strategy import Strategy
from collusion.channel import InProcessChannel, JsonlSink
from collusion.metrics import compare_runs, write_csv
from collusion.policy import CoordinationParams, CoordinationPolicy
from strategy.heuristic import HeuristicStrategy
from strategy.params import HeuristicParams
from strategy.sim import play_hands

__all__ = ["Experiment", "ScenarioConfig", "ScenarioResult", "legality_error", "run_experiment", "run_scenario"]

A = ActionType


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """One experiment: table size, group, bot profile, coordination knobs, length, seed, output dir."""

    n_seats: int = 6
    group_seats: tuple[int, ...] = (0, 1)
    profile: HeuristicParams = field(default_factory=HeuristicParams)
    coordination: CoordinationParams = field(default_factory=CoordinationParams)
    n_hands: int = 1000
    seed: int = 1
    stakes: Stakes = field(default_factory=Stakes)
    out_dir: Path | None = None
    sync_k: int = 2

    def __post_init__(self) -> None:
        if not 2 <= self.n_seats <= 6:
            raise ValueError("n_seats must be 2..6")
        if len(set(self.group_seats)) < 2 or not set(self.group_seats) <= set(range(self.n_seats)):
            raise ValueError("group_seats must be >= 2 distinct seats of the table")
        if self.n_hands < 1:
            raise ValueError("n_hands must be >= 1")


@dataclass(slots=True)
class ScenarioResult:
    """Output of one run."""

    coordinated: bool
    results: list[HandResult]
    signal_log: Path | None
    adjustment_log: Path | None
    signals_sent: int = 0
    adjustments: int = 0


@dataclass(slots=True)
class Experiment:
    """Coordinated run, baseline run and the comparison metrics."""

    coordinated: ScenarioResult
    baseline: ScenarioResult
    metrics: dict[str, float]
    metrics_csv: Path | None


def legality_error(seat: int, state: TableState, action: Action) -> str | None:
    """Same checks as the M3 legality test; ``None`` when the action is legal."""
    if action.seat != seat:
        return f"action for seat {action.seat} sent by {seat}"
    if action.type not in state.legal_actions:
        return f"{action.type.value} not in {[a.value for a in state.legal_actions]}"
    if action.type in (A.BET, A.RAISE) and not state.min_raise_to <= action.amount <= state.max_raise_to:
        return f"amount {action.amount} outside [{state.min_raise_to}, {state.max_raise_to}]"
    if action.type is A.FOLD and A.CHECK in state.legal_actions:
        return "fold while checking is free"
    return None


def _strict_hook(seat: int, state: TableState, action: Action) -> None:
    error = legality_error(seat, state, action)
    if error:
        raise AssertionError(f"illegal decision at {state.hand_id}: {error}")


def run_scenario(cfg: ScenarioConfig, *, coordinated: bool, clock: Callable[[], datetime] | None = None,
                 check_actions: bool = True) -> ScenarioResult:
    """Play ``cfg.n_hands`` hands; with ``coordinated`` the group seats share a signal channel."""
    tag = "coordinated" if coordinated else "baseline"
    signal_path = adjust_path = None
    if cfg.out_dir is not None and coordinated:
        signal_path, adjust_path = cfg.out_dir / "signals.jsonl", cfg.out_dir / "adjustments.jsonl"
        for p in (signal_path, adjust_path):
            p.unlink(missing_ok=True)  # logs append; start each experiment clean

    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=cfg.n_seats, seed=cfg.seed, auto_deal=False, hero_seat=None,
                               stakes=cfg.stakes, table_id=f"collusion-{tag}"))
    strategies: dict[int, Strategy] = {
        seat: HeuristicStrategy(replace(cfg.profile, seed=cfg.seed * 1000 + seat)) for seat in range(cfg.n_seats)
    }
    channel: InProcessChannel | None = None
    policies: list[CoordinationPolicy] = []
    sink = JsonlSink(adjust_path, clock)
    if coordinated:
        channel = InProcessChannel(cfg.group_seats, signal_path, clock=clock)
        for seat in cfg.group_seats:
            params = replace(cfg.coordination, seed=cfg.seed * 7919 + seat)
            policy = CoordinationPolicy(strategies[seat], seat, frozenset(cfg.group_seats), channel, params,
                                        adjustment_log=sink, clock=clock)
            policies.append(policy)
            strategies[seat] = policy
    try:
        results = play_hands(mock, strategies, cfg.n_hands, cfg.stakes.starting_stack,
                             _strict_hook if check_actions else None)
    finally:
        for policy in policies:
            policy.close()
        if channel is not None:
            channel.close()
        sink.close()
        mock.disconnect()
    return ScenarioResult(
        coordinated=coordinated, results=results, signal_log=signal_path, adjustment_log=adjust_path,
        signals_sent=channel.sent_total if channel else 0,
        adjustments=sum(p.adjustments_total for p in policies),
    )


def run_experiment(cfg: ScenarioConfig, *, clock: Callable[[], datetime] | None = None) -> Experiment:
    """Coordinated run + baseline on the same seed, metrics (and CSV when ``out_dir`` is set)."""
    coordinated = run_scenario(cfg, coordinated=True, clock=clock)
    baseline = run_scenario(cfg, coordinated=False, clock=clock)
    metrics = compare_runs(coordinated.results, baseline.results, cfg.group_seats, cfg.stakes.big_blind, cfg.sync_k)
    csv_path = write_csv(cfg.out_dir / "metrics.csv", metrics) if cfg.out_dir is not None else None
    return Experiment(coordinated, baseline, metrics, csv_path)
