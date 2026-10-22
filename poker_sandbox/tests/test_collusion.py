"""Tests for the M4 coordination layer (synthetic bots on the offline mock)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from adapter.config import AdapterConfig
from adapter.mock import MockAdapter
from adapter.models import RANK_CHARS, SUIT_CHARS, Action, ActionType, TableState
from agent.strategy import BaseStrategy
from collusion import (
    Bucket,
    CoordinationParams,
    CoordinationPolicy,
    InProcessChannel,
    Line,
    Message,
    ScenarioConfig,
    SignalIntent,
    SignalStrength,
    decode,
    encode,
    legality_error,
    make_channel,
    run_experiment,
    run_scenario,
)
from collusion.protocol import CARD_TOKEN
from pokermath import parse_many
from strategy import HeuristicParams, HeuristicStrategy

A = ActionType
C = parse_many
ALL_CARDS = [r + s for r in RANK_CHARS for s in SUIT_CHARS]
ALWAYS = CoordinationParams(coordination_strength=1.0, signal_interval=1, seed=1)


# ------------------------------------------------------------------ protocol


def _all_messages() -> list[Message]:
    msgs: list[Message] = []
    for seat in range(6):
        for hand_id in ("sandbox-1-000001", "collusion-coordinated-000042", "t-9"):
            msgs += [SignalStrength(seat, hand_id, b) for b in Bucket]
            msgs += [SignalIntent(seat, hand_id, line) for line in Line]
    return msgs


def test_serialized_messages_contain_no_cards() -> None:
    for msg in _all_messages():
        raw = encode(msg)
        text = raw.decode()
        assert decode(raw) == msg
        assert not CARD_TOKEN.search(text)
        data = json.loads(raw)
        assert not {"hole_cards", "board", "cards", "hand"} & data.keys()
        for card in ALL_CARDS:  # raw-byte check, independent of the protocol's own regex
            assert not re.search(rb"(?<![A-Za-z0-9])" + card.encode() + rb"(?![A-Za-z0-9])", raw)


def test_protocol_rejects_card_smuggling() -> None:
    with pytest.raises(ValueError):
        SignalStrength(0, "As-Kd", Bucket.STRONG)
    with pytest.raises(ValueError):
        SignalIntent(0, "h Ah", Line.I_PRESSURE)
    with pytest.raises(ValueError):
        SignalStrength(0, "h1", "huge")  # type: ignore[arg-type]
    good = json.loads(encode(SignalStrength(1, "h1", Bucket.WEAK)))
    for extra in ({"hole_cards": ["As", "Kd"]}, {"board": "2c3d4h"}, {"note": "x"}):
        with pytest.raises(ValueError):
            decode(json.dumps({**good, **extra}).encode())
    with pytest.raises(ValueError):
        decode(json.dumps({**good, "hand_id": "Qs"}).encode())


# ------------------------------------------------------------------- channel


def test_channel_delivers_only_to_group_partners(tmp_path: Path) -> None:
    log = tmp_path / "signals.jsonl"
    channel = InProcessChannel({0, 1, 2}, log)
    got: dict[int, list[Message]] = {0: [], 1: [], 2: []}
    for seat in (0, 1):
        channel.subscribe(seat, got[seat].append)
    msg = SignalStrength(0, "h1", Bucket.MONSTER)
    channel.send(0, msg)
    assert got == {0: [], 1: [msg], 2: []}
    channel.subscribe(2, got[2].append)  # backlog is delivered on subscribe
    assert got[2] == [msg]
    for bad in (lambda: channel.subscribe(3, print), lambda: channel.send(3, SignalStrength(3, "h1", Bucket.WEAK)),
                lambda: channel.send(1, msg)):
        with pytest.raises(ValueError):
            bad()
    channel.close()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["sender"], r["recipient"]) for r in records] == [(0, 1), (0, 2)]
    assert all({"ts", "sender", "recipient", "kind", "payload_hash", "payload"} <= r.keys() for r in records)
    with pytest.raises(RuntimeError):
        channel.send(0, msg)
    with pytest.raises(NotImplementedError):
        make_channel("redis", {0, 1})
    with pytest.raises(ValueError):
        make_channel("carrier-pigeon", {0, 1})


# -------------------------------------------------------------------- policy


class Fixed(BaseStrategy):
    """Always proposes the same action kind (min raise for BET/RAISE)."""

    def __init__(self, kind: ActionType) -> None:
        self.kind = kind

    def decide(self, state: TableState) -> Action:
        assert state.to_act_seat is not None
        amount = state.min_raise_to if self.kind in (A.BET, A.RAISE) else 0
        return Action(state.to_act_seat, self.kind, amount, hand_id=state.hand_id)


def _spot(hero_cards: str, utg_action: Action) -> tuple[MockAdapter, str]:
    """6-max, button 0: SB 1, BB 2, UTG 3 (partner), MP 4 (hero, to act)."""
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=6, seed=3, auto_deal=False, hero_seat=None))
    hand_id = mock.start_hand(button=0, hole_cards={4: C(hero_cards)})
    mock.send_action(utg_action)
    assert mock.view(None).to_act_seat == 4
    return mock, hand_id


def _policy(delegate: BaseStrategy, tmp_path: Path) -> tuple[CoordinationPolicy, InProcessChannel, Path]:
    channel = InProcessChannel({3, 4}, tmp_path / "signals.jsonl")
    log = tmp_path / "adjust.jsonl"
    return CoordinationPolicy(delegate, 4, {3, 4}, channel, ALWAYS, adjustment_log=log), channel, log


def _rules(log: Path) -> list[str]:
    return [json.loads(line)["rule_id"] for line in log.read_text().splitlines()] if log.exists() else []


def test_rule_a_yields_to_strong_partner(tmp_path: Path) -> None:
    mock, hid = _spot("9c 8c", Action(3, A.RAISE, 6))
    policy, channel, log = _policy(Fixed(A.RAISE), tmp_path)
    channel.send(3, SignalStrength(3, hid, Bucket.STRONG))
    action = policy.decide(mock.view(4))
    policy.close()
    assert action.type in (A.CALL, A.FOLD)
    assert _rules(log) == ["a_yield_to_strong"]
    record = json.loads(log.read_text())
    assert record["before_action"]["type"] == "raise" and record["partners_signals_seen"][0]["bucket"] == "strong"


def test_rule_b_no_overraise_with_medium(tmp_path: Path) -> None:
    mock, hid = _spot("Qs Js", Action(3, A.RAISE, 6))
    policy, channel, log = _policy(Fixed(A.RAISE), tmp_path)
    channel.send(3, SignalIntent(3, hid, Line.I_PRESSURE))
    action = policy.decide(mock.view(4))
    policy.close()
    assert action.type is A.CALL and _rules(log) == ["b_no_overraise"]


def test_rule_c_extracts_with_monster(tmp_path: Path) -> None:
    mock, hid = _spot("As Ah", Action(3, A.CALL))  # partner limps
    policy, channel, log = _policy(Fixed(A.CALL), tmp_path)
    channel.send(3, SignalStrength(3, hid, Bucket.WEAK))
    state = mock.view(4)
    action = policy.decide(state)
    policy.close()
    assert action.type is A.RAISE and action.amount == state.current_bet + state.pot + state.to_call
    assert _rules(log) == ["c_extract"]


def test_without_signals_policy_equals_delegate(tmp_path: Path) -> None:
    for i, hole in enumerate(["As Ah", "Qs Js", "9c 8c", "7h 2c", "Ks Qd", "5d 5c"]):
        mock, _ = _spot(hole, Action(3, A.RAISE, 6))
        state = mock.view(4)
        params = HeuristicParams(seed=100 + i)
        channel = InProcessChannel({3, 4})
        policy = CoordinationPolicy(HeuristicStrategy(params), 4, {3, 4}, channel, ALWAYS)
        assert policy.decide(state) == HeuristicStrategy(params).decide(state)
        assert policy.adjustments_total == 0
        policy.close()


def test_policy_stays_inside_its_group() -> None:
    channel = InProcessChannel({3, 4})
    with pytest.raises(ValueError):
        CoordinationPolicy(Fixed(A.CALL), 5, {4, 5}, channel)  # seat 5 not covered by the channel
    with pytest.raises(ValueError):
        CoordinationPolicy(Fixed(A.CALL), 2, {3, 4}, channel)  # seat outside its own group
    for bad in ({"coordination_strength": 1.5}, {"signal_interval": 0}, {"strength_samples": 1}):
        with pytest.raises(ValueError):
            CoordinationParams(**bad)


# ------------------------------------------------------------ runner/metrics


def _records_without_ts(path: Path) -> list[dict[str, object]]:
    out = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        rec.pop("ts")
        out.append(rec)
    return out


def test_determinism(tmp_path: Path) -> None:
    def run(sub: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        cfg = ScenarioConfig(n_hands=150, seed=4, coordination=ALWAYS, out_dir=tmp_path / sub)
        res = run_scenario(cfg, coordinated=True)
        assert res.signal_log is not None and res.adjustment_log is not None
        actions = [f"{r.hand_id}:{a.seat}:{a.type.value}:{a.amount}" for r in res.results for a in r.actions]
        return _records_without_ts(res.signal_log), _records_without_ts(res.adjustment_log), actions

    first, second = run("a"), run("b")
    assert first == second
    assert first[0], "no signals were exchanged"


def test_runner_actions_legal_and_logs_card_free(tmp_path: Path) -> None:
    exp = run_experiment(ScenarioConfig(n_hands=300, seed=2, coordination=ALWAYS, out_dir=tmp_path))
    assert len(exp.coordinated.results) == len(exp.baseline.results) == 300  # strict legality hook raised nothing
    assert exp.coordinated.signals_sent > 0
    assert exp.metrics_csv is not None and exp.metrics_csv.read_text().startswith("metric,value")
    assert exp.coordinated.signal_log is not None
    for line in exp.coordinated.signal_log.read_text().splitlines():
        rec = json.loads(line)
        assert not CARD_TOKEN.search(json.dumps(rec["payload"]))
        assert rec["sender"] in (0, 1) and rec["recipient"] in (0, 1) and rec["sender"] != rec["recipient"]
    state = TableState(table_id="t", max_seats=2, small_blind=1, big_blind=2, players=(), to_act_seat=0,
                       legal_actions=(A.CHECK, A.BET), min_raise_to=2, max_raise_to=10)
    assert legality_error(0, state, Action(0, A.FOLD)) is not None
    assert legality_error(0, state, Action(0, A.BET, 50)) is not None
    assert legality_error(0, state, Action(0, A.BET, 4)) is None


def test_group_metrics_point_the_right_way() -> None:
    exp = run_experiment(ScenarioConfig(n_hands=500, seed=1, coordination=ALWAYS))
    m = exp.metrics
    assert exp.coordinated.adjustments > 0
    assert m["sync_rate_group"] > m["sync_rate_baseline"]
    assert m["yield_rate_group"] > m["yield_rate_baseline"]
