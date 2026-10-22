"""CoordinationPolicy: wraps a Strategy and adjusts its action using partners' signals.

Rules are deliberately simple and explicit (labelled data for detectors, not EV):
  a) partner strong/monster and I want to bet/raise → check/call/fold (yield);
     a_soft_fold: partner strong/monster bet into me and I would call with ≤ medium → fold;
  b) partner signals I_pressure, I am medium and want to raise → call/check (no over-raise);
  c) I am a monster, partners are weak and I would check/call → pot-size raise (extract);
  d) no usable signals → the delegate's action unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.models import Action, ActionType, Event, EventType, HandResult, Street, TableState
from agent.strategy import BaseStrategy, Strategy, passive_or_fold
from collusion.channel import Clock, JsonlSink, SignalChannel
from collusion.protocol import Bucket, Line, Message, SignalIntent, SignalStrength, strength_signal

__all__ = ["CoordinationParams", "CoordinationPolicy", "PartnerSignal"]

A = ActionType
_AGGRESSIVE = (A.BET, A.RAISE, A.ALL_IN)


@dataclass(frozen=True, slots=True)
class CoordinationParams:
    """Knobs of the coordination layer.

    Attributes:
        coordination_strength: Probability that a matching rule is actually applied.
        signal_interval: Send signals at most once every N own decisions.
        strength_samples: Monte Carlo samples for the strength bucket.
        seed: RNG seed for rule mixing and equity sampling.
    """

    coordination_strength: float = 0.8
    signal_interval: int = 2
    strength_samples: int = 300
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.coordination_strength <= 1.0:
            raise ValueError("coordination_strength must be in [0, 1]")
        if self.signal_interval < 1:
            raise ValueError("signal_interval must be >= 1")
        if self.strength_samples < 50:
            raise ValueError("strength_samples must be >= 50")


@dataclass(frozen=True, slots=True)
class PartnerSignal:
    """Latest signals of one partner in the current hand."""

    seat: int
    bucket: Bucket | None = None
    line: Line | None = None

    def as_log(self) -> dict[str, Any]:
        """JSON-friendly view for the adjustment log."""
        return {"seat": self.seat, "bucket": self.bucket and self.bucket.value, "line": self.line and self.line.value}


def _action_log(action: Action) -> dict[str, Any]:
    return {"type": action.type.value, "amount": action.amount}


class CoordinationPolicy(BaseStrategy):
    """Delegating strategy for one seat of a coordinating group."""

    def __init__(
        self,
        delegate: Strategy,
        seat: int,
        group_seats: frozenset[int] | set[int] | tuple[int, ...],
        channel: SignalChannel,
        params: CoordinationParams | None = None,
        *,
        adjustment_log: JsonlSink | Path | str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.group_seats = frozenset(group_seats)
        if seat not in self.group_seats:
            raise ValueError("seat must belong to group_seats")
        if not self.group_seats <= channel.group:
            raise ValueError("channel group does not cover group_seats")
        self.delegate = delegate
        self.seat = seat
        self.partners = self.group_seats - {seat}
        self.channel = channel
        self.params = params or CoordinationParams()
        self._rng = random.Random(self.params.seed)
        self._sink = adjustment_log if isinstance(adjustment_log, JsonlSink) else JsonlSink(adjustment_log, clock)
        self._owns_sink = not isinstance(adjustment_log, JsonlSink)
        self._signals: dict[int, tuple[str, Bucket | None, Line | None]] = {}
        self._bucket_cache: dict[tuple[str, Street], Bucket] = {}
        self._decisions_since_signal = self.params.signal_interval  # allow a signal on the first decision
        self._street_aggressor: int | None = None
        self.adjustments_total = 0
        self.signals_sent = 0
        self._unsubscribe = channel.subscribe(seat, self._on_signal)

    @property
    def name(self) -> str:
        """``Coordinated(<delegate name>)``."""
        return f"Coordinated({self.delegate.name})"

    # ------------------------------------------------------------- hooks --

    def _on_signal(self, msg: Message) -> None:
        if msg.seat not in self.partners:
            return  # the channel never routes these, but stay strict
        hand_id, bucket, line = self._signals.get(msg.seat, (msg.hand_id, None, None))
        if hand_id != msg.hand_id:
            bucket, line = None, None
        if isinstance(msg, SignalStrength):
            bucket = msg.bucket
        else:
            line = msg.line
        self._signals[msg.seat] = (msg.hand_id, bucket, line)

    def on_event(self, event: Event) -> None:
        """Forward to the delegate; track who bet last on the current street."""
        if event.type in (EventType.HAND_STARTED, EventType.STREET_DEALT):
            self._street_aggressor = None
        elif event.type is EventType.ACTION and A(event.payload["type"]) in (A.BET, A.RAISE):
            self._street_aggressor = int(event.payload["seat"])
        self.delegate.on_event(event)

    def on_hand_end(self, result: HandResult) -> None:
        """Forward to the delegate and drop per-hand caches."""
        self._bucket_cache = {k: v for k, v in self._bucket_cache.items() if k[0] != result.hand_id}
        self.delegate.on_hand_end(result)

    def close(self) -> None:
        """Unsubscribe from the channel and close an owned log."""
        self._unsubscribe()
        if self._owns_sink:
            self._sink.close()

    # ---------------------------------------------------------- decisions --

    def decide(self, state: TableState) -> Action:
        """Delegate, maybe signal partners, then maybe adjust the delegate's action."""
        action = self.delegate.decide(state)
        if state.to_act_seat != self.seat or state.hand_id is None or state.street is None:
            return action
        hole = state.player(self.seat).hole_cards
        if not hole:
            return action
        my_bucket = self._bucket(state, hole)
        self._maybe_signal(state, my_bucket, action)
        partners = self._partner_signals(state)
        if not partners:
            return action  # rule d
        matched = self._match_rule(state, action, my_bucket, partners)
        if matched is None or self._rng.random() >= self.params.coordination_strength:
            return action
        adjusted, rule_id = matched
        self.adjustments_total += 1
        self._sink.write({"hand_id": state.hand_id, "seat": self.seat, "before_action": _action_log(action),
                          "after_action": _action_log(adjusted), "rule_id": rule_id,
                          "my_bucket": my_bucket.value,
                          "partners_signals_seen": [p.as_log() for p in partners]})
        return adjusted

    def _bucket(self, state: TableState, hole: tuple[Any, ...]) -> Bucket:
        assert state.hand_id is not None and state.street is not None
        key = (state.hand_id, state.street)
        if key not in self._bucket_cache:
            signal = strength_signal(self.seat, state.hand_id, hole, state.board,
                                     self.params.strength_samples, self._rng.getrandbits(31))
            self._bucket_cache[key] = signal.bucket
        return self._bucket_cache[key]

    def _maybe_signal(self, state: TableState, bucket: Bucket, planned: Action) -> None:
        self._decisions_since_signal += 1
        if self._decisions_since_signal < self.params.signal_interval:
            return
        live_partners = [s for s in self.partners if self._in_hand(state, s)]
        if not live_partners:
            return
        assert state.hand_id is not None
        if bucket.level >= Bucket.STRONG.level and planned.type in _AGGRESSIVE:
            line = Line.I_PRESSURE
        elif bucket is Bucket.WEAK:
            line = Line.I_FOLD_TO_YOU
        elif bucket is Bucket.MEDIUM and planned.type is A.CALL:
            line = Line.I_CALL_DOWN
        else:
            line = Line.I_YIELD if bucket is Bucket.MEDIUM else Line.I_PRESSURE
        self.channel.send(self.seat, SignalStrength(self.seat, state.hand_id, bucket))
        self.channel.send(self.seat, SignalIntent(self.seat, state.hand_id, line))
        self.signals_sent += 2
        self._decisions_since_signal = 0

    @staticmethod
    def _in_hand(state: TableState, seat: int) -> bool:
        try:
            p = state.player(seat)
        except KeyError:
            return False
        return p.in_hand and not p.has_folded

    def _partner_signals(self, state: TableState) -> list[PartnerSignal]:
        out = []
        for seat in sorted(self.partners):
            entry = self._signals.get(seat)
            if entry is None or entry[0] != state.hand_id or not self._in_hand(state, seat):
                continue
            out.append(PartnerSignal(seat, entry[1], entry[2]))
        return out

    def _match_rule(self, state: TableState, action: Action, mine: Bucket,
                    partners: list[PartnerSignal]) -> tuple[Action, str] | None:
        aggressive = action.type in _AGGRESSIVE
        partner_strong = any(p.bucket is not None and p.bucket.level >= Bucket.STRONG.level for p in partners)
        pressure = any(p.line is Line.I_PRESSURE for p in partners)
        partner_weak = not partner_strong and any(
            p.bucket is Bucket.WEAK or p.line in (Line.I_YIELD, Line.I_FOLD_TO_YOU) for p in partners)

        if partner_strong and aggressive:
            if state.to_call == 0:
                return self._fit(state, A.CHECK), "a_yield_to_strong"
            kind = A.CALL if mine.level >= Bucket.MEDIUM.level else A.FOLD
            return self._fit(state, kind), "a_yield_to_strong"
        if (partner_strong and action.type is A.CALL and self._street_aggressor in self.partners
                and mine.level <= Bucket.MEDIUM.level):
            return self._fit(state, A.FOLD), "a_soft_fold"
        if pressure and mine is Bucket.MEDIUM and aggressive:
            return self._fit(state, A.CALL if state.to_call else A.CHECK), "b_no_overraise"
        if mine is Bucket.MONSTER and partner_weak and action.type in (A.CHECK, A.CALL):
            target = state.current_bet + state.pot + state.to_call  # pot-size raise
            fitted = self._fit(state, A.RAISE, target)
            if fitted.type in (A.BET, A.RAISE):
                return fitted, "c_extract"
        return None

    @staticmethod
    def _fit(state: TableState, kind: ActionType, amount: int = 0) -> Action:
        """Closest legal action to ``kind``/``amount``."""
        legal = state.legal_actions
        seat = state.to_act_seat
        assert seat is not None
        if kind in (A.BET, A.RAISE):
            aggressive = A.BET if A.BET in legal else A.RAISE if A.RAISE in legal else None
            if aggressive is not None:
                amount = max(state.min_raise_to, min(amount, state.max_raise_to))
                return Action(seat, aggressive, amount, hand_id=state.hand_id)
            kind = A.CALL
        if kind is A.CALL and A.CALL not in legal:
            kind = A.CHECK
        if kind is A.FOLD and A.CHECK in legal:
            kind = A.CHECK
        if kind not in legal:
            return passive_or_fold(state)
        return Action(seat, kind, hand_id=state.hand_id)
