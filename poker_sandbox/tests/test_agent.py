"""Tests for the M1 agent layer. Plain pytest: each test drives its own asyncio.run()."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from adapter.audit import JsonlAuditLog
from adapter.base import EventCallback, HandNotFoundError, IllegalActionError, PokerAdapter, Unsubscribe
from adapter.config import AdapterConfig
from adapter.mock import MockAdapter
from adapter.models import (
    Action,
    ActionType,
    Event,
    EventType,
    HandResult,
    PlayerState,
    Street,
    TableState,
    Visibility,
    cards,
)
from agent import (
    Agent,
    AgentConfig,
    AlwaysFoldStrategy,
    BaseStrategy,
    CallingStationStrategy,
    RandomStrategy,
    SeatBoundAdapter,
    Strategy,
    TransientAdapterError,
    state_hash,
)

A = ActionType
WAIT_S = 5.0
FAST = AgentConfig(decision_timeout_s=1.0, retry_base_s=0.001, retry_cap_s=0.004)

# ------------------------------------------------------------------ helpers


def turn_state(hand_id: str, *, to_call: int = 2, street: Street = Street.PREFLOP, seat: int = 0) -> TableState:
    legal = (A.FOLD, A.CALL, A.RAISE, A.ALL_IN) if to_call else (A.CHECK, A.BET, A.ALL_IN)
    return TableState(
        table_id="t", max_seats=2, small_blind=1, big_blind=2,
        players=(PlayerState(0, "p0", 100, hole_cards=cards("Ah Kh")), PlayerState(1, "p1", 100)),
        button_seat=0, hand_id=hand_id, hand_number=1, street=street, current_bet=to_call,
        to_act_seat=seat, legal_actions=legal, to_call=to_call, min_raise_to=4, max_raise_to=100,
    )


def idle_state() -> TableState:
    return TableState(table_id="t", max_seats=2, small_blind=1, big_blind=2,
                      players=(PlayerState(0, "p0", 100), PlayerState(1, "p1", 100)))


def hand_result(hand_id: str) -> HandResult:
    now = datetime.now(UTC)
    return HandResult(
        hand_id=hand_id, hand_number=1, table_id="t", button_seat=0, small_blind_seat=0, big_blind_seat=1,
        board=(), starting_stacks={0: 100, 1: 100}, final_stacks={0: 100, 1: 100}, blinds={}, actions=(),
        pots=(), winnings={}, net={0: 0, 1: 0}, revealed_cards={}, hand_ranks={}, went_to_showdown=False,
        started_at=now, ended_at=now,
    )


class FakeAdapter(PokerAdapter):
    """Scripted single-table adapter for seat 0; no filtering of events (tests do that)."""

    def __init__(self, script: list[TableState], *, auto_turn: bool = True,
                 fail_sends: int = 0, fail_exc: type[Exception] = ConnectionError) -> None:
        self.script = script
        self.idx = 0
        self.state = idle_state()
        self.auto_turn = auto_turn
        self.fail_sends = fail_sends
        self.fail_exc = fail_exc
        self.send_attempts = 0
        self.sent: list[Action] = []
        self.finished = threading.Event()
        self.subs: dict[int, EventCallback] = {}
        self._seq = 0
        self._next_sub = 0

    # PokerAdapter
    def connect(self, config: AdapterConfig) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    def view(self, seat: int | None) -> TableState:
        return self.state

    def get_table_state(self) -> TableState:
        return self.state

    def on_event(self, callback: EventCallback, *, seat: int | None = None) -> Unsubscribe:
        sub = self._next_sub
        self._next_sub += 1
        self.subs[sub] = callback
        return lambda: self.subs.pop(sub, None) and None

    def get_hand_history(self, hand_id: str) -> HandResult:
        if not hand_id.startswith("h"):
            raise HandNotFoundError(hand_id)
        return hand_result(hand_id)

    def send_action(self, action: Action) -> None:
        self.send_attempts += 1
        if self.fail_sends > 0:
            self.fail_sends -= 1
            raise self.fail_exc("transient")
        st = self.state
        if st.to_act_seat != action.seat or action.type not in st.legal_actions:
            raise IllegalActionError(f"illegal {action}")
        if action.type is A.RAISE and action.amount < st.min_raise_to:
            raise IllegalActionError("below min raise")
        self.sent.append(action)
        self._advance()

    # scripting
    def emit(self, etype: EventType, hand_id: str | None = None, payload: dict[str, Any] | None = None,
             visibility: Visibility = Visibility.PUBLIC, recipient: int | None = None) -> None:
        self._seq += 1
        event = Event(etype, datetime.now(UTC), self._seq, "t", hand_id, payload or {}, visibility, recipient)
        for cb in list(self.subs.values()):
            cb(event)

    def start(self) -> None:
        self.state = self.script[0]
        if self.auto_turn:
            self.emit(EventType.TURN, self.state.hand_id, {"seat": 0})

    def _advance(self) -> None:
        old = self.state.hand_id
        self.idx += 1
        if self.idx >= len(self.script):
            self.state = idle_state()
            self.emit(EventType.HAND_ENDED, old)
            self.finished.set()
            return
        self.state = self.script[self.idx]
        if self.state.hand_id != old:
            self.emit(EventType.HAND_ENDED, old)
        if self.auto_turn:
            self.emit(EventType.TURN, self.state.hand_id, {"seat": 0})


class Scripted(BaseStrategy):
    """CALL/CHECK normally; blocks on ``gate`` at indices in ``block_at``; illegal raise at ``illegal_at``."""

    def __init__(self, block_at: Iterable[int] = (), illegal_at: Iterable[int] = ()) -> None:
        self.calls = 0
        self.block_at = frozenset(block_at)
        self.illegal_at = frozenset(illegal_at)
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.events: list[Event] = []
        self.hands_ended: list[str] = []
        self.marker = threading.Event()

    def decide(self, state: TableState) -> Action:
        i = self.calls
        self.calls += 1
        assert state.to_act_seat is not None
        if i in self.block_at:
            self.entered.set()
            self.gate.wait(WAIT_S)
        if i in self.illegal_at:
            return Action(state.to_act_seat, A.RAISE, 1)
        return CallingStationStrategy().decide(state)

    def on_event(self, event: Event) -> None:
        self.events.append(event)
        if event.payload.get("marker"):
            self.marker.set()

    def on_hand_end(self, result: HandResult) -> None:
        self.hands_ended.append(result.hand_id)


async def wait_thread_event(ev: threading.Event) -> None:
    assert await asyncio.to_thread(ev.wait, WAIT_S), "timed out waiting"


async def start(agent: Agent) -> asyncio.Task[None]:
    task = asyncio.create_task(agent.run())
    await asyncio.wait_for(agent.wait_started(), WAIT_S)
    return task


async def finish(agent: Agent, task: asyncio.Task[None]) -> None:
    await asyncio.wait_for(agent.stop(), WAIT_S)
    await asyncio.wait_for(task, WAIT_S)


def bound(fake: FakeAdapter) -> SeatBoundAdapter:
    return SeatBoundAdapter(fake, 0)


# --------------------------------------------------------------- lifecycle


def test_protocol_conformance() -> None:
    for s in (RandomStrategy(1), AlwaysFoldStrategy(), CallingStationStrategy(), Scripted()):
        assert isinstance(s, Strategy)
    assert AlwaysFoldStrategy().name == "AlwaysFoldStrategy"
    with pytest.raises(TypeError):
        Agent(bound(FakeAdapter([])), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Agent(FakeAdapter([]), AlwaysFoldStrategy())


def test_stop_ends_loop_and_unsubscribes() -> None:
    async def scenario() -> None:
        fake = FakeAdapter([])
        agent = Agent(bound(fake), AlwaysFoldStrategy(), FAST)
        task = await start(agent)
        assert agent.running and len(fake.subs) == 1
        await finish(agent, task)
        assert task.done() and not agent.running and fake.subs == {}
        await agent.stop()  # idempotent

    asyncio.run(scenario())


def test_stop_waits_for_current_decision() -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1")])
        strat = Scripted(block_at={0})
        agent = Agent(bound(fake), strat, FAST)
        task = await start(agent)
        await asyncio.to_thread(fake.start)
        await wait_thread_event(strat.entered)
        stopping = asyncio.create_task(agent.stop())
        await asyncio.sleep(0)
        assert not stopping.done()
        strat.gate.set()
        await asyncio.wait_for(stopping, WAIT_S)
        await task
        assert [a.type for a in fake.sent] == [A.CALL]
        assert agent.metrics.decisions_total == 1

    asyncio.run(scenario())


# ------------------------------------------------- timeout / retry / illegal


@pytest.mark.parametrize("to_call, expected", [(2, A.FOLD), (0, A.CHECK)])
def test_timeout_sends_legal_fallback(to_call: int, expected: ActionType, tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1", to_call=to_call)])
        strat = Scripted(block_at={0})
        cfg = FAST.model_copy(update={"decision_timeout_s": 0.05, "decision_log_path": tmp_path / "d.jsonl"})
        agent = Agent(bound(fake), strat, cfg)
        task = await start(agent)
        try:
            await asyncio.to_thread(fake.start)
            await wait_thread_event(fake.finished)
        finally:
            strat.gate.set()  # release the abandoned worker thread
        await finish(agent, task)
        assert [a.type for a in fake.sent] == [expected]
        assert agent.metrics.timeouts_total == 1 and agent.metrics.decisions_total == 1
        rec = json.loads((tmp_path / "d.jsonl").read_text())
        assert rec["reason"] == "timeout" and rec["action"]["type"] == expected.value

    asyncio.run(scenario())


@pytest.mark.parametrize("exc", [ConnectionError, TimeoutError, OSError, TransientAdapterError])
def test_transient_errors_are_retried(exc: type[Exception]) -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1")], fail_sends=2, fail_exc=exc)
        agent = Agent(bound(fake), CallingStationStrategy(), FAST)
        task = await start(agent)
        await asyncio.to_thread(fake.start)
        await wait_thread_event(fake.finished)
        await finish(agent, task)
        assert fake.send_attempts == 3
        assert agent.metrics.retries_total == 2
        assert agent.metrics.decisions_total == 1 and agent.metrics.errors_total == 0

    asyncio.run(scenario())


def test_retries_give_up_after_retry_max() -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1")], fail_sends=10)
        agent = Agent(bound(fake), CallingStationStrategy(), FAST)
        task = await start(agent)
        await asyncio.to_thread(fake.start)
        for _ in range(200):  # wait for the 4th attempt to fail and be counted
            if agent.metrics.errors_total:
                break
            await asyncio.sleep(0.005)
        await finish(agent, task)
        assert fake.send_attempts == 1 + FAST.retry_max
        assert agent.metrics.retries_total == FAST.retry_max
        assert agent.metrics.errors_total == 1 and agent.metrics.decisions_total == 0

    asyncio.run(scenario())


def test_illegal_action_not_retried_and_folds(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1")])
        cfg = FAST.model_copy(update={"decision_log_path": tmp_path / "d.jsonl"})
        agent = Agent(bound(fake), Scripted(illegal_at={0}), cfg)
        task = await start(agent)
        await asyncio.to_thread(fake.start)
        await wait_thread_event(fake.finished)
        await finish(agent, task)
        assert fake.send_attempts == 2  # the illegal raise + the fold, no retries
        assert [a.type for a in fake.sent] == [A.FOLD]
        m = agent.metrics
        assert (m.errors_total, m.retries_total, m.decisions_total) == (1, 0, 1)
        assert json.loads((tmp_path / "d.jsonl").read_text())["reason"] == "illegal_fallback"

    asyncio.run(scenario())


# ------------------------------------------------------------------ metrics


def test_metrics_scenario_two_hands() -> None:
    """2 hands x 5 normal decisions + 1 timeout + 1 illegal = 12 sent actions."""

    async def scenario() -> None:
        script = [turn_state("h1")] * 6 + [turn_state("h2", to_call=0)] * 6
        fake = FakeAdapter(script)
        strat = Scripted(block_at={2}, illegal_at={8})
        cfg = FAST.model_copy(update={"decision_timeout_s": 0.05})
        agent = Agent(bound(fake), strat, cfg)
        task = await start(agent)
        try:
            await asyncio.to_thread(fake.start)
            await wait_thread_event(fake.finished)
        finally:
            strat.gate.set()
        await finish(agent, task)
        m = agent.metrics
        assert m.decisions_total == 12 == len(fake.sent)
        assert (m.timeouts_total, m.errors_total, m.retries_total) == (1, 1, 0)
        # h1: 5 calls + timeout fold; h2 (check-free): 5 checks + illegal raise → fold illegal → check
        assert m.decisions_by_action == {"call": 5, "fold": 1, "check": 6}
        assert len(m.decision_latency_ms) == 12 and m.latency_percentile_ms(95) >= m.latency_percentile_ms(50)
        assert strat.hands_ended == ["h1", "h2"]
        snap = m.snapshot()
        assert snap["decisions_total"] == 12 and snap["latency_ms"]["mean"] > 0

    asyncio.run(scenario())


# ------------------------------------------------------------- backpressure


def test_backpressure_drops_oldest_and_resyncs(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeAdapter([turn_state("h1"), turn_state("h1", street=Street.FLOP)], auto_turn=False)
        strat = Scripted(block_at={0})
        cfg = FAST.model_copy(update={"queue_maxsize": 4, "decision_log_path": tmp_path / "d.jsonl"})
        agent = Agent(bound(fake), strat, cfg)
        task = await start(agent)
        await asyncio.to_thread(fake.start)
        agent._enqueue(Event(EventType.TURN, datetime.now(UTC), 0, "t", "h1", {"seat": 0}))
        await wait_thread_event(strat.entered)  # loop is now busy deciding; nobody drains the queue

        def burst() -> None:
            fake.emit(EventType.TURN, "h1", {"seat": 0})  # the real flop TURN — will be dropped
            for i in range(9):
                fake.emit(EventType.ACTION, "h1", {"seat": 1, "filler": i})

        await asyncio.to_thread(burst)
        strat.gate.set()
        await wait_thread_event(fake.finished)
        await finish(agent, task)
        m = agent.metrics
        assert m.events_dropped_total == 6  # 10 events into a queue of 4
        assert m.resyncs_total == 1
        assert m.decisions_total == 2  # the dropped TURN was recovered by the resync
        reasons = [json.loads(line)["reason"] for line in (tmp_path / "d.jsonl").read_text().splitlines()]
        assert reasons == ["strategy", "resync"]
        assert not any(e.payload.get("filler") in range(5) for e in strat.events)  # oldest went first

    asyncio.run(scenario())


# --------------------------------------------------------- event visibility


def test_second_line_drops_foreign_events() -> None:
    async def scenario() -> None:
        fake = FakeAdapter([])
        strat = Scripted()
        agent = Agent(bound(fake), strat, FAST)
        task = await start(agent)

        def leak() -> None:
            fake.emit(EventType.DECK, "h1", {"seed": 1}, Visibility.AUDIT)
            fake.emit(EventType.HOLE_CARDS, "h1", {"seat": 1}, Visibility.PRIVATE, recipient=1)
            fake.emit(EventType.HOLE_CARDS, "h1", {"seat": 0}, Visibility.PRIVATE, recipient=0)
            fake.emit(EventType.ACTION, "h1", {"marker": True})

        await asyncio.to_thread(leak)
        await wait_thread_event(strat.marker)
        await finish(agent, task)
        assert agent.metrics.foreign_events_total == 2
        assert [(e.type, e.recipient_seat) for e in strat.events] == [
            (EventType.HOLE_CARDS, 0), (EventType.ACTION, None)]

    asyncio.run(scenario())


def run_mock_table(
    seats: int, hands: int, tmp_path: Path, make_strategy: Callable[[int], Scripted]
) -> tuple[MockAdapter, dict[int, Agent], dict[int, Scripted]]:
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=seats, seed=5, auto_deal=False, hero_seat=None,
                               log_path=tmp_path / "audit.jsonl"))
    wrappers = SeatBoundAdapter.for_seats(mock, range(seats))
    strategies = {s: make_strategy(s) for s in range(seats)}
    agents = {s: Agent(wrappers[s], strategies[s], FAST) for s in range(seats)}

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        def on_hand_end(event: Event) -> None:
            if event.type is EventType.HAND_ENDED:
                loop.call_soon_threadsafe(done.set)

        unsub = wrappers[0].run_locked(lambda: mock.on_event(on_hand_end))
        tasks = [asyncio.create_task(a.run()) for a in agents.values()]
        await asyncio.gather(*(a.wait_started() for a in agents.values()))
        for _ in range(hands):
            done.clear()
            await asyncio.to_thread(wrappers[0].run_locked, mock.start_hand)
            await asyncio.wait_for(done.wait(), WAIT_S)
        # stop() drops queued events by design: let every agent see the last HAND_ENDED first.
        async def all_hands_seen() -> None:
            while any(len(st.hands_ended) < hands for st in strategies.values()):
                await asyncio.sleep(0.001)

        await asyncio.wait_for(all_hands_seen(), WAIT_S)
        await asyncio.gather(*(a.stop() for a in agents.values()))
        await asyncio.gather(*tasks)
        wrappers[0].run_locked(unsub)

    asyncio.run(scenario())
    return mock, agents, strategies


def test_agent_gets_only_public_and_own_private_events(tmp_path: Path) -> None:
    mock, agents, strategies = run_mock_table(6, 3, tmp_path, lambda s: Scripted())
    for seat, strat in strategies.items():
        assert strat.events, seat
        assert all(e.visibility is not Visibility.AUDIT for e in strat.events)
        private = [e for e in strat.events if e.visibility is Visibility.PRIVATE]
        assert private and all(e.recipient_seat == seat for e in private)
        assert sum(e.type is EventType.HOLE_CARDS for e in strat.events) == 3
        assert agents[seat].metrics.foreign_events_total == 0
        assert len(strat.hands_ended) == 3
    # The service events exist (in the M0 audit log) — they just never reached an agent.
    audit_types = {r["type"] for r in JsonlAuditLog.read(tmp_path / "audit.jsonl")}
    assert {"deck", "session_started"} <= audit_types
    played = sum(len(mock.get_hand_history(h).actions) for h in mock.hand_ids)
    assert played == sum(a.metrics.decisions_total for a in agents.values())


def test_state_hash_ignores_other_seats_hole_cards() -> None:
    def seat0_state(other_cards: str) -> TableState:
        mock = MockAdapter()
        mock.connect(AdapterConfig(seats=2, seed=9, auto_deal=False, hero_seat=0))
        mock.start_hand(button=0, hole_cards={0: cards("Ah Kh"), 1: cards(other_cards)})
        return SeatBoundAdapter(mock, 0).get_table_state()

    a, b = seat0_state("2c 2d"), seat0_state("Qs Qd")
    assert a == b and state_hash(a, 0) == state_hash(b, 0)
    leaked = replace(a, players=tuple(replace(p, hole_cards=cards("Qs Qd")) if p.seat == 1 else p
                                      for p in a.players))
    assert state_hash(leaked, 0) == state_hash(a, 0)
    own_changed = replace(a, players=tuple(replace(p, hole_cards=cards("2s 3s")) if p.seat == 0 else p
                                           for p in a.players))
    assert state_hash(own_changed, 0) != state_hash(a, 0)


def test_seat_bound_adapter_guards() -> None:
    fake = FakeAdapter([turn_state("h1")])
    wrapper = SeatBoundAdapter(fake, 0)
    with pytest.raises(IllegalActionError):
        wrapper.send_action(Action(1, A.FOLD))
    with pytest.raises(ValueError):
        wrapper.on_event(lambda e: None, seat=1)
    with pytest.raises(TypeError):
        SeatBoundAdapter(object(), 0)  # type: ignore[arg-type]
    assert wrapper.view_for_detector() == fake.view(0)


def test_baseline_strategies_pick_legal_actions() -> None:
    facing = turn_state("h1", to_call=2)
    free = turn_state("h1", to_call=0)
    assert AlwaysFoldStrategy().decide(facing).type is A.FOLD
    assert AlwaysFoldStrategy().decide(free).type is A.CHECK
    assert CallingStationStrategy().decide(facing).type is A.CALL
    assert CallingStationStrategy().decide(free).type is A.CHECK
    r1, r2 = RandomStrategy(3), RandomStrategy(3)
    seq1 = [r1.decide(facing) for _ in range(50)]
    assert seq1 == [r2.decide(facing) for _ in range(50)]
    assert {a.type for a in seq1} <= set(facing.legal_actions)
    assert all(facing.min_raise_to <= a.amount <= facing.max_raise_to for a in seq1 if a.type is A.RAISE)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import socket

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
