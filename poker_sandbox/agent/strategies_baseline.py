"""Baseline strategies used to calibrate detectors. Deliberately naive."""

from __future__ import annotations

import random

from adapter.models import Action, ActionType, TableState
from agent.strategy import BaseStrategy, fold_or_passive, passive_or_fold


class RandomStrategy(BaseStrategy):
    """Uniform over ``legal_actions``; BET/RAISE size uniform in [min_raise_to, max_raise_to]."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(self, state: TableState) -> Action:
        """Pick a random legal action."""
        if state.to_act_seat is None:
            raise ValueError("nobody is to act")
        kind = self._rng.choice(state.legal_actions)
        amount = 0
        if kind in (ActionType.BET, ActionType.RAISE):
            amount = self._rng.randint(state.min_raise_to, state.max_raise_to)
        return Action(state.to_act_seat, kind, amount, hand_id=state.hand_id)


class AlwaysFoldStrategy(BaseStrategy):
    """FOLD if legal, otherwise CHECK/CALL."""

    def decide(self, state: TableState) -> Action:
        """Fold whenever possible."""
        return fold_or_passive(state)


class CallingStationStrategy(BaseStrategy):
    """CHECK/CALL if legal, otherwise FOLD."""

    def decide(self, state: TableState) -> Action:
        """Never raise, never fold when calling is possible."""
        return passive_or_fold(state)
