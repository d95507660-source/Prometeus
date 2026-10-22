"""Seat-bound view of a shared table adapter.

A local table (e.g. the M0 mock) is driven by several agents at once, while
``PokerAdapter.get_table_state`` only shows the configured hero. This wrapper
binds one seat, serialises every call to the shared, single-threaded inner
adapter with one lock, and subscribes with ``seat=self.seat`` so the inner
adapter's own filtering delivers public + this seat's private events only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar, runtime_checkable

from adapter.base import EventCallback, IllegalActionError, PokerAdapter, Unsubscribe
from adapter.config import AdapterConfig
from adapter.models import Action, HandResult, TableState

T = TypeVar("T")


@runtime_checkable
class SupportsSeatView(Protocol):
    """Adapter that can render the table from an arbitrary seat."""

    def view(self, seat: int | None) -> TableState:
        """Snapshot as seen by ``seat``."""
        ...


class SeatBoundAdapter(PokerAdapter):
    """:class:`PokerAdapter` for one seat of a shared inner adapter.

    Callbacks run in whichever thread made the inner call (under the lock); they
    must not call back into the adapter from another thread while blocking on it.
    The agent's callbacks only hand events to the event loop, so no deadlock.
    """

    def __init__(self, inner: PokerAdapter, seat: int, lock: threading.RLock | None = None) -> None:
        if not isinstance(inner, SupportsSeatView):
            raise TypeError("inner adapter must implement view(seat)")
        self._inner = inner
        self._viewer: SupportsSeatView = inner
        self._seat = seat
        # RLock: inner callbacks fire in the calling thread, which already holds it.
        self._lock = lock if lock is not None else threading.RLock()

    @classmethod
    def for_seats(cls, inner: PokerAdapter, seats: Iterable[int]) -> dict[int, SeatBoundAdapter]:
        """Wrappers for several seats sharing one lock."""
        lock = threading.RLock()
        return {seat: cls(inner, seat, lock) for seat in seats}

    @property
    def seat(self) -> int:
        """Bound seat."""
        return self._seat

    @property
    def inner(self) -> PokerAdapter:
        """Wrapped adapter; call it only through :meth:`run_locked`."""
        return self._inner

    def run_locked(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` under the shared lock (for drivers: start_hand, set_stack, …)."""
        with self._lock:
            return fn()

    def connect(self, config: AdapterConfig) -> None:
        """Connect the inner adapter."""
        with self._lock:
            self._inner.connect(config)

    def disconnect(self) -> None:
        """Disconnect the inner adapter (affects every seat sharing it)."""
        with self._lock:
            self._inner.disconnect()

    @property
    def is_connected(self) -> bool:
        """Inner connection state."""
        with self._lock:
            return self._inner.is_connected

    def get_table_state(self) -> TableState:
        """Table as seen by the bound seat (own hole cards only)."""
        with self._lock:
            return self._viewer.view(self._seat)

    def view_for_detector(self) -> TableState:
        """TODO(M6/M7): full view with every seat's hole cards needs a server-side API.

        Stub: returns ``inner.view(seat)`` unchanged. Reserved for the future
        detector branch; must not be used for decisions.
        """
        with self._lock:
            return self._viewer.view(self._seat)

    def send_action(self, action: Action) -> None:
        """Send an action for the bound seat only."""
        if action.seat != self._seat:
            raise IllegalActionError(f"adapter is bound to seat {self._seat}, got action for {action.seat}")
        with self._lock:
            self._inner.send_action(action)

    def on_event(self, callback: EventCallback, *, seat: int | None = None) -> Unsubscribe:
        """Subscribe with the bound seat's visibility; ``seat`` must be ``None`` or the bound seat."""
        if seat not in (None, self._seat):
            raise ValueError(f"adapter is bound to seat {self._seat}")
        with self._lock:
            unsubscribe = self._inner.on_event(callback, seat=self._seat)

        def locked_unsubscribe() -> None:
            with self._lock:
                unsubscribe()

        return locked_unsubscribe

    def get_hand_history(self, hand_id: str) -> HandResult:
        """History of a finished hand."""
        with self._lock:
            return self._inner.get_hand_history(hand_id)
