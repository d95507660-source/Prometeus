"""Asyncio decision loop on top of a synchronous :class:`PokerAdapter`.

Adapter calls run via ``asyncio.to_thread``; adapter callbacks only hand events
to the loop with ``call_soon_threadsafe``. All agent state (queue, metrics,
logger) is touched from the loop thread only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Final, TypeVar

from adapter.base import HandNotFoundError, IllegalActionError, PokerAdapter, Unsubscribe
from adapter.models import Event, EventType, TableState, Visibility
from agent.config import AgentConfig
from agent.logging import DecisionLogger
from agent.metrics import AgentMetrics
from agent.seat import SeatBoundAdapter
from agent.strategy import Strategy, fold_or_passive

T = TypeVar("T")

log = logging.getLogger(__name__)


class TransientAdapterError(Exception):
    """Raise from an adapter for errors worth retrying (network blips, 5xx, …)."""


TRANSIENT_ERRORS: Final = (TransientAdapterError, ConnectionError, TimeoutError, OSError)


class _Stop:
    """Queue sentinel that wakes an idle ``run()`` on ``stop()``."""


_STOP: Final = _Stop()


class Agent:
    """Plays one seat: waits for its TURN, asks the strategy, sends the action.

    Args:
        adapter: Usually a :class:`SeatBoundAdapter`; any adapter works if ``seat``
            is given and ``get_table_state`` shows that seat.
        strategy: Decision policy.
        config: Loop tuning; defaults to :class:`AgentConfig`.
        seat: Seat to play; taken from ``adapter.seat`` for seat-bound adapters.
        decision_logger: Shared logger; otherwise one is created from
            ``config.decision_log_path`` and closed when ``run()`` ends.
    """

    def __init__(
        self,
        adapter: PokerAdapter,
        strategy: Strategy,
        config: AgentConfig | None = None,
        *,
        seat: int | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy does not implement the Strategy protocol")
        if seat is None:
            if not isinstance(adapter, SeatBoundAdapter):
                raise ValueError("seat is required for adapters that are not seat-bound")
            seat = adapter.seat
        self._adapter = adapter
        self._strategy = strategy
        self._config = config or AgentConfig()
        self._seat = seat
        self._owns_logger = decision_logger is None
        self._logger = decision_logger
        self.metrics = AgentMetrics()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Event | _Stop] | None = None
        self._unsubscribe: Unsubscribe | None = None
        self._stopping = False
        self._resync_pending = False
        self._started = asyncio.Event()
        self._done = asyncio.Event()
        self._running = False

    @property
    def seat(self) -> int:
        """Seat this agent plays."""
        return self._seat

    @property
    def running(self) -> bool:
        """True between subscription in ``run()`` and its exit."""
        return self._running

    async def wait_started(self) -> None:
        """Wait until ``run()`` has subscribed to events."""
        await self._started.wait()

    async def run(self) -> None:
        """Process events until :meth:`stop`. Must be awaited on the loop that calls ``stop``."""
        if self._running:
            raise RuntimeError("agent is already running")
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._config.queue_maxsize)
        self._stopping = False
        self._resync_pending = False
        self._done.clear()
        if self._owns_logger:
            self._logger = DecisionLogger(self._config.decision_log_path)
        self._unsubscribe = await asyncio.to_thread(self._adapter.on_event, self._on_adapter_event, seat=self._seat)
        self._running = True
        try:
            await self._guarded(self._check_turn(None, "strategy"))  # it may already be our turn
            self._started.set()
            while not self._stopping:
                item = await self._queue.get()
                if isinstance(item, _Stop) or self._stopping:
                    break
                if self._resync_pending:
                    self._resync_pending = False
                    self.metrics.resyncs_total += 1
                    await self._guarded(self._check_turn(None, "resync"))
                await self._guarded(self._handle(item))
        finally:
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            if unsubscribe is not None:
                await asyncio.to_thread(unsubscribe)
            if self._owns_logger and self._logger is not None:
                self._logger.close()
            self._running = False
            self._started.clear()
            self._done.set()

    async def stop(self) -> None:
        """Finish the current decision, unsubscribe and let ``run()`` return. Does not disconnect."""
        if not self._running:
            return
        self._stopping = True
        self._enqueue(_STOP)
        await self._done.wait()

    # ------------------------------------------------------------ events --

    def _on_adapter_event(self, event: Event) -> None:
        """Adapter thread → loop. Never blocks the adapter."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # loop closed between the check and the call
            loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, item: Event | _Stop) -> None:
        queue = self._queue
        if queue is None:
            return
        if queue.full():
            dropped = queue.get_nowait()
            if not isinstance(dropped, _Stop):
                self.metrics.events_dropped_total += 1
                self._resync_pending = True
                log.warning("seat %s: queue full, dropped event seq=%s type=%s",
                            self._seat, dropped.seq, dropped.type.value)
        queue.put_nowait(item)

    def _is_foreign(self, event: Event) -> bool:
        if event.visibility is Visibility.AUDIT:
            return True
        return event.visibility is Visibility.PRIVATE and event.recipient_seat != self._seat

    async def _guarded(self, step: Awaitable[None]) -> None:
        """Await a step; any unexpected error is counted and logged, never kills the loop."""
        try:
            await step
        except Exception:
            self.metrics.errors_total += 1
            log.exception("seat %s: agent step failed", self._seat)

    async def _handle(self, event: Event) -> None:
        if self._is_foreign(event):
            self.metrics.foreign_events_total += 1
            log.error("seat %s: foreign %s event %s dropped", self._seat, event.visibility.value, event.type.value)
            return
        self._strategy.on_event(event)
        if event.type is EventType.TURN and event.payload.get("seat") == self._seat:
            await self._check_turn(event.hand_id, "strategy")
        elif event.type is EventType.HAND_ENDED and event.hand_id is not None:
            try:
                result = await self._adapter_call(self._adapter.get_hand_history, event.hand_id)
            except HandNotFoundError:
                log.warning("seat %s: no history for %s", self._seat, event.hand_id)
                return
            self._strategy.on_hand_end(result)

    # --------------------------------------------------------- decisions --

    async def _check_turn(self, expected_hand_id: str | None, reason: str) -> None:
        """Re-read the state and decide only if it is really our turn in the expected hand."""
        state = await self._adapter_call(self._adapter.get_table_state)
        if state.to_act_seat != self._seat or state.hand_id is None:
            return
        if expected_hand_id is not None and state.hand_id != expected_hand_id:
            return  # stale TURN from a finished hand
        await self._decide_and_send(state, reason)

    async def _decide_and_send(self, state: TableState, reason: str) -> None:
        started = time.perf_counter()
        try:
            action = await asyncio.wait_for(
                asyncio.to_thread(self._strategy.decide, state), self._config.decision_timeout_s
            )
        except TimeoutError:
            # The worker thread cannot be cancelled; its late result is discarded.
            self.metrics.timeouts_total += 1
            log.warning("seat %s: decide() timed out in %s", self._seat, state.hand_id)
            action, reason = fold_or_passive(state), "timeout"
        except Exception:
            # A crashing strategy is a strategy bug: same treatment as an illegal action.
            self.metrics.errors_total += 1
            log.exception("seat %s: decide() raised", self._seat)
            action, reason = fold_or_passive(state), "illegal_fallback"
        latency_ms = (time.perf_counter() - started) * 1000
        if action.hand_id is None:
            action = replace(action, hand_id=state.hand_id)

        try:
            await self._adapter_call(self._adapter.send_action, action)
        except IllegalActionError as exc:
            # Never retried: retrying a buggy decision would loop forever.
            self.metrics.errors_total += 1
            log.warning("seat %s: illegal %s rejected (%s); falling back", self._seat, action, exc)
            action, reason = fold_or_passive(state), "illegal_fallback"
            try:
                await self._adapter_call(self._adapter.send_action, action)
            except IllegalActionError:
                self.metrics.errors_total += 1
                log.error("seat %s: fallback %s rejected too; skipping turn", self._seat, action)
                return
        self.metrics.record_decision(action.type.value, latency_ms)
        if self._logger is not None:
            self._logger.log(
                state=state, seat=self._seat, action=action, latency_ms=latency_ms,
                strategy_name=self._strategy.name, reason=reason,
            )

    async def _adapter_call(self, fn: Callable[..., T], *args: object) -> T:
        """``to_thread`` with exponential backoff on transient errors only."""
        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(fn, *args)
            except IllegalActionError:
                raise
            except TRANSIENT_ERRORS as exc:
                if attempt >= self._config.retry_max:
                    raise
                delay = min(self._config.retry_cap_s, self._config.retry_base_s * 2**attempt)
                attempt += 1
                self.metrics.retries_total += 1
                log.info("seat %s: transient %r, retry %d in %.3fs", self._seat, exc, attempt, delay)
                await asyncio.sleep(delay)
