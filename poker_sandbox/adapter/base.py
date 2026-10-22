"""Abstract contract between bots and a poker sandbox.

Bots depend only on :class:`PokerAdapter` and the models in :mod:`adapter.models`;
swapping :class:`~adapter.mock.MockAdapter` for a real API client must not
require changes in bot logic.

The contract is synchronous and single-threaded: callbacks registered with
:meth:`PokerAdapter.on_event` are invoked in event order and may call
:meth:`PokerAdapter.send_action` re-entrantly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .config import AdapterConfig
from .models import Action, Event, HandResult, TableState

EventCallback = Callable[[Event], None]
Unsubscribe = Callable[[], None]


class AdapterError(Exception):
    """Base class for adapter errors."""


class NotConnectedError(AdapterError):
    """Raised when an operation requires an active connection."""


class IllegalActionError(AdapterError):
    """Raised when an action violates the rules or is sent out of turn."""


class HandNotFoundError(AdapterError, KeyError):
    """Raised when a hand history is unknown or the hand is still running."""


class TableClosedError(AdapterError):
    """Raised when a new hand cannot be started (not enough players, hand limit)."""


class PokerAdapter(ABC):
    """Interface every sandbox backend implements."""

    @abstractmethod
    def connect(self, config: AdapterConfig) -> None:
        """Open a session with the given configuration."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the session and flush the audit log. Idempotent."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether :meth:`connect` succeeded and :meth:`disconnect` was not called."""

    @abstractmethod
    def get_table_state(self) -> TableState:
        """Return the current table snapshot as seen by ``config.hero_seat``."""

    @abstractmethod
    def send_action(self, action: Action) -> None:
        """Submit an action.

        Raises:
            IllegalActionError: the action is not legal right now. The rejection
                is also published as an ``ACTION_REJECTED`` event and logged.
        """

    @abstractmethod
    def on_event(self, callback: EventCallback, *, seat: int | None = None) -> Unsubscribe:
        """Subscribe to table events.

        Args:
            callback: Called once per event, in ``seq`` order.
            seat: Whose private events (e.g. hole cards) this subscriber
                receives. ``None`` means ``config.hero_seat``. Audit-only events
                are never delivered.

        Returns:
            A function that removes the subscription.
        """

    @abstractmethod
    def get_hand_history(self, hand_id: str) -> HandResult:
        """Return the history of a finished hand.

        Raises:
            HandNotFoundError: unknown ``hand_id`` or hand still in progress.
        """

    @contextmanager
    def session(self, config: AdapterConfig) -> Iterator[PokerAdapter]:
        """``with adapter.session(cfg) as a: ...`` — connect, then always disconnect."""
        self.connect(config)
        try:
            yield self
        finally:
            self.disconnect()
