"""Strategy protocol: the only thing a bot author implements.

Strategies are plain synchronous objects. They never touch the adapter: they get
an immutable :class:`TableState` and return an :class:`Action`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adapter.models import Action, ActionType, Event, HandResult, TableState


@runtime_checkable
class Strategy(Protocol):
    """Decision policy for one seat."""

    @property
    def name(self) -> str:
        """Identifier written to the decision log."""
        ...

    def decide(self, state: TableState) -> Action:
        """Return the action for ``state.to_act_seat``. Runs in a worker thread."""
        ...

    def on_event(self, event: Event) -> None:
        """Observe a table event (public or own private). Runs on the event loop: keep it cheap."""
        ...

    def on_hand_end(self, result: HandResult) -> None:
        """Observe the history of a finished hand."""
        ...


class BaseStrategy:
    """Mixin with no-op hooks and a class-name ``name``; subclasses implement ``decide``."""

    @property
    def name(self) -> str:
        """Defaults to the class name."""
        return type(self).__name__

    def decide(self, state: TableState) -> Action:
        """Must be overridden."""
        raise NotImplementedError

    def on_event(self, event: Event) -> None:
        """No-op."""

    def on_hand_end(self, result: HandResult) -> None:
        """No-op."""


def _first_legal(state: TableState, order: tuple[ActionType, ...]) -> Action:
    if state.to_act_seat is None:
        raise ValueError("nobody is to act")
    for kind in order:
        if kind in state.legal_actions:
            return Action(state.to_act_seat, kind, hand_id=state.hand_id)
    raise ValueError(f"none of {order} is legal in {state.legal_actions}")


def fold_or_passive(state: TableState) -> Action:
    """FOLD if legal, otherwise the first legal of CHECK/CALL (the agent's safe fallback)."""
    return _first_legal(state, (ActionType.FOLD, ActionType.CHECK, ActionType.CALL))


def passive_or_fold(state: TableState) -> Action:
    """CHECK/CALL if legal, otherwise FOLD."""
    return _first_legal(state, (ActionType.CHECK, ActionType.CALL, ActionType.FOLD))
