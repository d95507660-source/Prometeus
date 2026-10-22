"""HumanLikeWrapper: wraps any Strategy with planned timings, random "human" errors and tilt.

The delegate is never modified or patched: its action is taken as is and possibly replaced
by a legal error action. Timings are only planned (logged and exposed via
``last_planned_delay_ms``); nothing sleeps.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol, runtime_checkable

from adapter.models import Action, ActionType, Event, HandResult, TableState
from agent.strategy import BaseStrategy, Strategy
from behavior.logging import BehaviorLogger, BehaviorRecord, Clock
from behavior.profiles import ErrorKind, HumanLikeProfile
from behavior.timing import TimingModel, context_from_state

__all__ = ["HumanLikeWrapper", "SupportsBluffBonus", "SupportsEquity", "error_action"]

A = ActionType


@runtime_checkable
class SupportsBluffBonus(Protocol):
    """Delegate that accepts an additive bluff-frequency bonus."""

    bluff_bonus: float


@runtime_checkable
class SupportsEquity(Protocol):
    """Delegate that exposes the equity of its last decision."""

    last_equity: float | None


def error_action(kind: ErrorKind, state: TableState, base: Action) -> Action | None:
    """Legal action for an error of ``kind``, or ``None`` if it is impossible or equals ``base``."""
    seat = state.to_act_seat
    if seat is None:
        return None
    legal = state.legal_actions
    if kind == "random_fold":
        # Folding when checking is free is not a plausible error (and M3's legality check forbids it).
        if A.FOLD in legal and A.CHECK not in legal and base.type is not A.FOLD:
            return Action(seat, A.FOLD, hand_id=state.hand_id)
        return None
    if kind == "random_call":
        passive = A.CALL if A.CALL in legal else A.CHECK if A.CHECK in legal else None
        if passive is not None and base.type is not passive:
            return Action(seat, passive, hand_id=state.hand_id)
        return None
    aggressive = A.BET if A.BET in legal else A.RAISE if A.RAISE in legal else None
    if aggressive is None or state.min_raise_to > state.max_raise_to:
        return None
    if base.type is aggressive and base.amount == state.min_raise_to:
        return None
    return Action(seat, aggressive, state.min_raise_to, hand_id=state.hand_id)


class HumanLikeWrapper(BaseStrategy):
    """Delegating strategy that adds human-like noise; ``humanlike_level == 0`` is a transparent pass-through."""

    def __init__(
        self,
        delegate: Strategy,
        profile: HumanLikeProfile | None = None,
        *,
        seed: int | None = None,
        seat: int | None = None,
        behavior_log: BehaviorLogger | Path | str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.delegate = delegate
        self.profile = profile or HumanLikeProfile()
        self.seat = seat
        self._rng = random.Random(seed)  # own RNG: the delegate's stream is never touched
        self._logger = behavior_log if isinstance(behavior_log, BehaviorLogger) else BehaviorLogger(behavior_log, clock)
        self._owns_logger = not isinstance(behavior_log, BehaviorLogger)
        self._base_bluff_bonus = delegate.bluff_bonus if isinstance(delegate, SupportsBluffBonus) else 0.0
        self._big_blind: int | None = None
        self.loss_streak_bb = 0.0
        self.hands_seen = 0
        self.tilt_hand_ids: set[str] = set()
        self.records: list[BehaviorRecord] = []
        self.last_planned_delay_ms: float | None = None

    @property
    def name(self) -> str:
        """``HumanLike(<delegate name>)``."""
        return f"HumanLike({self.delegate.name})"

    @property
    def bluff_bonus_supported(self) -> bool:
        """Whether the delegate exposes ``bluff_bonus`` (otherwise tilt bluffing is ignored)."""
        return isinstance(self.delegate, SupportsBluffBonus)

    @property
    def tilted(self) -> bool:
        """On tilt: enough big blinds lost over consecutive losing hands (never at level 0)."""
        return self.profile.humanlike_level > 0 and self.loss_streak_bb >= self.profile.tilt_threshold

    @property
    def session_pos(self) -> float:
        """Share of the planned session already played (0..1)."""
        return min(1.0, self.hands_seen / self.profile.session_hands)

    @property
    def errors_total(self) -> int:
        """Decisions replaced by an error action."""
        return sum(r.error_applied for r in self.records)

    # ------------------------------------------------------------- hooks --

    def on_event(self, event: Event) -> None:
        """Forward to the delegate."""
        self.delegate.on_event(event)

    def on_hand_end(self, result: HandResult) -> None:
        """Forward, then update the losing streak (in bb) that drives tilt."""
        self.delegate.on_hand_end(result)
        self.hands_seen += 1
        if self.seat is None or self.seat not in result.net:
            return
        bb = self._big_blind or max(result.blinds.values(), default=1)
        net = result.net[self.seat]
        if net < 0:
            self.loss_streak_bb += -net / bb
        elif net > 0:
            self.loss_streak_bb = 0.0  # any winning hand ends the streak; break-even hands keep it

    def decide(self, state: TableState) -> Action:
        """Delegate's action, possibly replaced by an error; records a planned delay."""
        seat = state.to_act_seat
        if seat is None:
            raise ValueError("nobody is to act")
        if self.seat is None:
            self.seat = seat
        self._big_blind = state.big_blind
        tilted = self.tilted
        self._sync_bluff_bonus(tilted)
        base = self.delegate.decide(state)
        final, kind = self._maybe_error(state, base, tilted)
        equity = self.delegate.last_equity if isinstance(self.delegate, SupportsEquity) else None
        ctx = context_from_state(state, session_pos=self.session_pos, tilted=tilted, equity=equity)
        delay = TimingModel.sample(self.profile, ctx, self._rng)
        self.last_planned_delay_ms = delay
        record = BehaviorRecord(
            hand_id=state.hand_id, seat=seat, street=state.street.value if state.street else None,
            base_action=base, final_action=final, error_applied=kind is not None, error_kind=kind,
            planned_delay_ms=delay, tilt_state=tilted, humanlike_level=self.profile.humanlike_level,
        )
        self.records.append(record)
        self._logger.log(record)
        if tilted and state.hand_id is not None:
            self.tilt_hand_ids.add(state.hand_id)
        return final

    def close(self) -> None:
        """Close the behaviour log if this wrapper opened it."""
        if self._owns_logger:
            self._logger.close()

    # ----------------------------------------------------------- helpers --

    def _sync_bluff_bonus(self, tilted: bool) -> None:
        if isinstance(self.delegate, SupportsBluffBonus):
            extra = self.profile.humanlike_level * self.profile.tilt_bluff_bonus if tilted else 0.0
            self.delegate.bluff_bonus = self._base_bluff_bonus + extra

    def _maybe_error(self, state: TableState, base: Action, tilted: bool) -> tuple[Action, ErrorKind | None]:
        p = self.profile
        u, pick = self._rng.random(), self._rng.random()  # always two draws: aligned streams across levels
        rate = p.humanlike_level * p.error_rate * (p.tilt_error_mult if tilted else 1.0)
        if u >= rate:
            return base, None
        candidates = [(k, a) for k in p.error_kinds if (a := error_action(k, state, base)) is not None]
        if not candidates:
            return base, None
        kind, action = candidates[min(int(pick * len(candidates)), len(candidates) - 1)]
        return action, kind
