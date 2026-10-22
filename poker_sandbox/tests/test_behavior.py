"""Tests for the M5 behaviour layer (sync self-play on the M0 mock, synthetic bots only)."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from adapter.config import AdapterConfig
from adapter.mock import MockAdapter
from adapter.models import Action, HandResult, Street, TableState
from agent.strategies_baseline import CallingStationStrategy
from agent.strategy import BaseStrategy, Strategy
from behavior import (
    PRESETS,
    ROBOT,
    TILTING_HUMAN,
    BehaviorLogger,
    DecisionContext,
    HumanLikeProfile,
    HumanLikeWrapper,
    TimingModel,
    timing_report,
)
from collusion.runner import legality_error
from strategy import HeuristicParams, HeuristicStrategy, play_hands

STACK = 200  # 100 bb
FAST = HeuristicParams(equity_samples=100)  # keeps self-play quick; behaviour does not depend on it
NO_TILT = 1e9


def table(seats: int = 6, seed: int = 1) -> MockAdapter:
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=seats, seed=seed, auto_deal=False, hero_seat=None))
    return mock


def heuristic(seats: int, base_seed: int) -> dict[int, HeuristicStrategy]:
    return {s: HeuristicStrategy(replace(FAST, seed=base_seed + s)) for s in range(seats)}


def wrapped(seats: int, profile: HumanLikeProfile, base_seed: int,
            log: BehaviorLogger | None = None) -> dict[int, HumanLikeWrapper]:
    return {s: HumanLikeWrapper(d, profile, seed=base_seed * 7 + s, seat=s, behavior_log=log)
            for s, d in heuristic(seats, base_seed).items()}


def all_records(wrappers: dict[int, HumanLikeWrapper]) -> list[dict[str, object]]:
    return [r.as_log() for w in wrappers.values() for r in w.records]


def recorded_states(hands: int, seed: int = 4) -> tuple[list[TableState], list[HandResult]]:
    states: list[TableState] = []
    results = play_hands(table(6, seed), heuristic(6, seed), hands, STACK,
                         lambda seat, st, a: states.append(st))
    return states, results


def contexts(n: int, seed: int = 0) -> list[DecisionContext]:
    rng = random.Random(seed)
    streets = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
    out = []
    for i in range(n):
        pot = rng.uniform(1.5, 60)
        out.append(DecisionContext(street=rng.choice(streets), pot_bb=pot, to_call_bb=rng.choice([0, pot / 2, pot]),
                                   is_close_decision=rng.random() < 0.3, session_pos=i / n))
    return out


def sample_delays(profile: HumanLikeProfile, ctxs: list[DecisionContext], seed: int) -> list[float]:
    rng = random.Random(seed)
    return [TimingModel.sample(profile, c, rng) for c in ctxs]


# ----------------------------------------------------------- profiles


def test_profile_validation_and_presets() -> None:
    for bad in ({"base_latency_ms": 0}, {"latency_sigma": -1}, {"error_rate": 0.2}, {"humanlike_level": 1.5},
                {"tank_prob": -0.1}, {"complexity_factor": 0.5}, {"error_kinds": ("nope",)},
                {"error_kinds": ()}, {"session_hands": 0}, {"tilt_threshold": float("nan")}):
        with pytest.raises(ValueError):
            HumanLikeProfile(**bad)
    assert ROBOT.humanlike_level == 0.0
    assert all(p.humanlike_level > 0 for name, p in PRESETS.items() if name != "robot")


# ------------------------------------------------------------- timing


def test_timing_level0_is_constant() -> None:
    profile = replace(HumanLikeProfile(), humanlike_level=0.0)
    assert set(sample_delays(profile, contexts(500), 1)) == {profile.base_latency_ms}


def test_timing_level1_noise_and_determinism() -> None:
    ctxs = contexts(4000)
    robot = timing_report(sample_delays(replace(HumanLikeProfile(), humanlike_level=0.0), ctxs, 1))
    human = timing_report(sample_delays(HumanLikeProfile(), ctxs, 1))
    assert human.mean > robot.mean and human.p95 > robot.p95 and human.p99 >= human.p95 >= human.p50
    assert robot.cv == 0.0 and human.cv > 0.3
    assert sample_delays(HumanLikeProfile(), ctxs, 5) == sample_delays(HumanLikeProfile(), ctxs, 5)
    assert sample_delays(HumanLikeProfile(), ctxs, 5) != sample_delays(HumanLikeProfile(), ctxs, 6)


def test_fatigue_and_complexity_slow_down() -> None:
    profile = replace(HumanLikeProfile(), latency_sigma=0.0, tank_prob=0.0)
    rng = random.Random(0)
    easy = TimingModel.sample(profile, DecisionContext(), rng)
    hard = TimingModel.sample(profile, DecisionContext(Street.RIVER, 40, 40, True), rng)
    tired = TimingModel.sample(profile, DecisionContext(session_pos=1.0), rng)
    assert easy == profile.base_latency_ms < tired < hard


# ------------------------------------------------------------ wrapper


def _flow(results: list[HandResult]) -> list[tuple[str, int, str, int]]:
    return [(r.hand_id, a.seat, a.type.value, a.amount) for r in results for a in r.actions]  # no timestamps


def test_level0_is_transparent_over_500_hands() -> None:
    plain = play_hands(table(6, 2), heuristic(6, 30), 500, STACK)
    wrappers = wrapped(6, ROBOT, 30)
    robot = play_hands(table(6, 2), wrappers, 500, STACK)
    assert _flow(plain) == _flow(robot)
    records = [r for w in wrappers.values() for r in w.records]
    assert len(records) > 2000
    assert all(not r.error_applied and r.final_action == r.base_action and not r.tilt_state for r in records)
    assert {r.planned_delay_ms for r in records} == {ROBOT.base_latency_ms}


def test_level1_error_rate_and_kinds() -> None:
    profile = replace(HumanLikeProfile(), error_rate=0.08, tilt_threshold=NO_TILT)
    wrappers = wrapped(6, profile, 40)
    play_hands(table(6, 3), wrappers, 600, STACK)
    records = [r for w in wrappers.values() for r in w.records]
    errors = [r for r in records if r.error_applied]
    assert len(records) > 2500
    assert abs(len(errors) / len(records) - 0.08) <= 0.03
    assert all(r.final_action != r.base_action for r in errors)
    kinds = Counter(r.error_kind for r in errors)
    assert set(kinds) == {"random_fold", "random_call", "min_raise"}
    assert min(kinds.values()) >= 0.05 * len(errors)  # random_fold is rarer: often already folded/free check


def test_tilt_raises_errors_and_delays() -> None:
    states, results = recorded_states(300)
    loss = next(r for r in results if r.net[0] <= -40)  # ≥ 20 bb lost by seat 0
    win = next(r for r in results if r.net[0] > 0)
    calm = HumanLikeWrapper(CallingStationStrategy(), TILTING_HUMAN, seed=11, seat=0)
    tilted = HumanLikeWrapper(CallingStationStrategy(), TILTING_HUMAN, seed=11, seat=0)
    tilted.on_hand_end(loss)
    assert tilted.tilted and not calm.tilted and tilted.loss_streak_bb >= TILTING_HUMAN.tilt_threshold
    for st in states:
        calm.decide(st)
        tilted.decide(st)
    calm_t = timing_report([r.planned_delay_ms for r in calm.records])
    tilt_t = timing_report([r.planned_delay_ms for r in tilted.records])
    assert tilted.errors_total > 1.5 * calm.errors_total
    assert tilt_t.mean > calm_t.mean and tilt_t.p95 > calm_t.p95 and tilt_t.cv > calm_t.cv
    assert all(r.tilt_state for r in tilted.records) and not any(r.tilt_state for r in calm.records)
    tilted.on_hand_end(win)
    assert not tilted.tilted


class _Bluffer(BaseStrategy):
    def __init__(self) -> None:
        self.bluff_bonus = 0.0
        self.seen: list[float] = []

    def decide(self, state: TableState) -> Action:
        self.seen.append(self.bluff_bonus)
        return CallingStationStrategy().decide(state)


def test_bluff_bonus_only_on_tilt_and_only_if_supported() -> None:
    states, results = recorded_states(20)
    loss = next(r for r in results if r.net[0] < 0)
    delegate = _Bluffer()
    w = HumanLikeWrapper(delegate, replace(TILTING_HUMAN, tilt_threshold=0.1), seat=0)
    w.decide(states[0])
    w.on_hand_end(loss)
    w.decide(states[1])
    assert w.bluff_bonus_supported and delegate.seen == [0.0, TILTING_HUMAN.tilt_bluff_bonus]
    assert not HumanLikeWrapper(HeuristicStrategy(), TILTING_HUMAN).bluff_bonus_supported
    assert isinstance(w, Strategy)


def _stripped(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all("ts" in r for r in rows)
    return [{k: v for k, v in r.items() if k != "ts"} for r in rows]


def test_determinism_same_seed(tmp_path: Path) -> None:
    def run(seed: int, name: str) -> tuple[list[dict[str, object]], Path]:
        path = tmp_path / name
        with BehaviorLogger(path) as log:
            wrappers = wrapped(6, TILTING_HUMAN, seed, log)
            play_hands(table(6, 5), wrappers, 200, STACK)
        return all_records(wrappers), path

    a, a_log = run(7, "a.jsonl")
    b, b_log = run(7, "b.jsonl")
    c, _ = run(8, "c.jsonl")
    assert [r["final_action"] for r in a] == [r["final_action"] for r in b]
    assert _stripped(a_log) == _stripped(b_log)
    assert a != c
    assert set(_stripped(a_log)[0]) == {"hand_id", "seat", "street", "base_action", "final_action",
                                        "error_applied", "error_kind", "planned_delay_ms", "tilt_state",
                                        "humanlike_level"}


def test_never_illegal_over_1000_hands() -> None:
    checked = 0

    def check(seat: int, state: TableState, action: Action) -> None:
        nonlocal checked
        checked += 1
        assert legality_error(seat, state, action) is None, (state, action)
        assert action.amount - state.player(seat).bet <= state.player(seat).stack or action.amount == 0

    profile = replace(TILTING_HUMAN, error_rate=0.15)  # max errors: stress the error actions
    hands = 0
    for seats, n, seed in ((6, 700, 50), (2, 300, 60)):
        wrappers = wrapped(seats, profile, seed)
        hands += len(play_hands(table(seats, seed), wrappers, n, STACK, check))
        assert sum(w.errors_total for w in wrappers.values()) > 0
    assert hands == 1000 and checked > 3000


def test_spectrum_is_monotone() -> None:
    states, _ = recorded_states(300)
    base = replace(HumanLikeProfile(), error_rate=0.1, tilt_threshold=NO_TILT)
    cvs, error_shares = [], []
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = HumanLikeWrapper(CallingStationStrategy(), replace(base, humanlike_level=level), seed=3)
        for st in states:
            w.decide(st)
        cvs.append(timing_report([r.planned_delay_ms for r in w.records]).cv)
        error_shares.append(w.errors_total / len(w.records))
    assert cvs == sorted(cvs) and len(set(cvs)) == 5, cvs
    assert error_shares == sorted(error_shares) and len(set(error_shares)) == 5, error_shares
    assert cvs[0] == 0.0 and error_shares[0] == 0.0


def test_logger_fixed_clock(tmp_path: Path) -> None:
    clock = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731
    states, _ = recorded_states(2)
    path = tmp_path / "b.jsonl"
    w = HumanLikeWrapper(CallingStationStrategy(), behavior_log=path, clock=clock, seed=1)
    w.decide(states[0])
    w.close()
    w.close()  # idempotent
    row = json.loads(path.read_text())
    assert row["ts"] == "2026-01-01T00:00:00+00:00"
    assert w.last_planned_delay_ms == pytest.approx(row["planned_delay_ms"], abs=1e-3)
