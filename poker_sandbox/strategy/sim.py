"""Synchronous local self-play driver (tests, examples). No asyncio, no network.

Works with any table exposing the M0 mock's local hooks; the mock is not imported here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from adapter.base import EventCallback, Unsubscribe
from adapter.models import Action, HandResult, TableState
from agent.strategy import Strategy

DecisionHook = Callable[[int, TableState, Action], None]


class LocalTable(Protocol):
    """Subset of MockAdapter used by :func:`play_hands`."""

    def view(self, seat: int | None) -> TableState:
        """Table seen by ``seat``."""
        ...

    def start_hand(self) -> str:
        """Deal the next hand."""
        ...

    def set_stack(self, seat: int, chips: int) -> None:
        """Reset a stack between hands."""
        ...

    def send_action(self, action: Action) -> None:
        """Apply an action."""
        ...

    def on_event(self, callback: EventCallback, *, seat: int | None = None) -> Unsubscribe:
        """Subscribe with ``seat``'s visibility."""
        ...

    def get_hand_history(self, hand_id: str) -> HandResult:
        """History of a finished hand."""
        ...


def play_hands(
    table: LocalTable,
    strategies: Mapping[int, Strategy],
    hands: int,
    stack: int,
    on_decision: DecisionHook | None = None,
) -> list[HandResult]:
    """Play ``hands`` hands with stacks reset to ``stack``; each seat's strategy sees only its own view."""
    unsubscribe = [table.on_event(s.on_event, seat=seat) for seat, s in strategies.items()]
    results: list[HandResult] = []
    try:
        for _ in range(hands):
            for seat in strategies:
                table.set_stack(seat, stack)
            hand_id = table.start_hand()
            while (seat_to_act := table.view(None).to_act_seat) is not None:
                view = table.view(seat_to_act)
                action = strategies[seat_to_act].decide(view)
                if on_decision is not None:
                    on_decision(seat_to_act, view, action)
                table.send_action(action)
            result = table.get_hand_history(hand_id)
            for s in strategies.values():
                s.on_hand_end(result)
            results.append(result)
    finally:
        for unsub in unsubscribe:
            unsub()
    return results


def seat_profiles(names: Sequence[str], seats: int) -> dict[int, str]:
    """Spread profile names round-robin over seats: 0→names[0], 1→names[1], …"""
    return {seat: names[seat % len(names)] for seat in range(seats)}
