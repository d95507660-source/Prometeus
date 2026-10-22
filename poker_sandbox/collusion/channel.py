"""Signal channel between the seats of one group, with a mandatory JSONL audit log.

Only an in-process backend exists: the sandbox runs in one process and must
not touch the network, so ``backend="redis"`` is refused (see :func:`make_channel`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from collusion.protocol import Message, decode, encode

__all__ = ["InProcessChannel", "JsonlSink", "MessageCallback", "SignalChannel", "make_channel"]

MessageCallback = Callable[[Message], None]
Clock = Callable[[], datetime]
log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JsonlSink:
    """Append-only JSONL writer; ``path=None`` is a no-op. Thread-safe."""

    def __init__(self, path: Path | str | None, clock: Clock | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock or _utcnow
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Write ``{"ts": ..., **record}`` as one line."""
        if self._fh is None:
            return
        line = json.dumps({"ts": self._clock().isoformat(), **record}, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the file. Idempotent."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


class SignalChannel(ABC):
    """Broadcast within a fixed group: a message from seat S goes to every other group seat."""

    group: frozenset[int]

    @abstractmethod
    def send(self, sender_seat: int, msg: Message) -> None:
        """Send ``msg`` from ``sender_seat`` (must be in the group and equal ``msg.seat``)."""

    @abstractmethod
    def subscribe(self, seat: int, callback: MessageCallback) -> Callable[[], None]:
        """Receive messages addressed to ``seat`` (group members only); returns unsubscribe."""

    @abstractmethod
    def close(self) -> None:
        """Stop accepting messages and flush the log."""


class InProcessChannel(SignalChannel):
    """Per-recipient ``queue.SimpleQueue`` + synchronous pump (thread-safe, re-entrancy safe).

    Messages for a seat without subscribers stay queued and are delivered on subscribe.
    """

    def __init__(self, group_seats: Iterable[int], log_path: Path | str | None = None,
                 *, clock: Clock | None = None) -> None:
        self.group = frozenset(group_seats)
        if len(self.group) < 2:
            raise ValueError("a group needs at least two seats")
        self._queues: dict[int, queue.SimpleQueue[Message]] = {s: queue.SimpleQueue() for s in self.group}
        self._subs: dict[int, dict[int, MessageCallback]] = {s: {} for s in self.group}
        self._next_id = 0
        self._lock = threading.RLock()
        self._pumping = False
        self._closed = False
        self.sink = JsonlSink(log_path, clock)
        self.sent_total = 0

    def send(self, sender_seat: int, msg: Message) -> None:
        """Validate, log one line per recipient, enqueue and deliver."""
        if self._closed:
            raise RuntimeError("channel is closed")
        if sender_seat not in self.group:
            raise ValueError(f"seat {sender_seat} is not in the group")
        if msg.seat != sender_seat:
            raise ValueError("message seat must equal the sender")
        raw = encode(msg)  # raises if anything card-like would leave
        payload = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        with self._lock:
            self.sent_total += 1
            for recipient in sorted(self.group - {sender_seat}):
                self.sink.write({"sender": sender_seat, "recipient": recipient, "kind": payload["kind"],
                                 "payload_hash": digest, "payload": payload})
                self._queues[recipient].put(decode(raw))  # recipients get a decoded copy, never the object
            self._pump()

    def subscribe(self, seat: int, callback: MessageCallback) -> Callable[[], None]:
        """Register ``callback`` for messages addressed to ``seat``."""
        if seat not in self.group:
            raise ValueError(f"seat {seat} is not in the group")
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subs[seat][sub_id] = callback
            self._pump()

        def unsubscribe() -> None:
            with self._lock:
                self._subs[seat].pop(sub_id, None)

        return unsubscribe

    async def asend(self, sender_seat: int, msg: Message) -> None:
        """Async-friendly wrapper (delivery is in-memory and never blocks)."""
        self.send(sender_seat, msg)

    def _pump(self) -> None:
        if self._pumping:
            return  # a callback sent a message; the outer loop delivers it
        self._pumping = True
        try:
            progress = True
            while progress:
                progress = False
                for seat in sorted(self.group):
                    q, subs = self._queues[seat], list(self._subs[seat].values())
                    while subs and not q.empty():
                        msg = q.get_nowait()
                        progress = True
                        for cb in subs:
                            try:
                                cb(msg)
                            except Exception:
                                log.exception("signal callback for seat %s failed", seat)
        finally:
            self._pumping = False

    def close(self) -> None:
        """Refuse further sends and close the log."""
        self._closed = True
        self.sink.close()


def make_channel(backend: str, group_seats: Iterable[int], log_path: Path | str | None = None,
                 *, clock: Clock | None = None) -> SignalChannel:
    """Channel factory. Only ``"memory"`` is available.

    ``"redis"`` is deliberately not provided: the bench is single-process and
    network-free, so a cross-process transport has no use here.
    """
    if backend == "memory":
        return InProcessChannel(group_seats, log_path, clock=clock)
    if backend == "redis":
        raise NotImplementedError(
            "redis backend is not available in this sandbox (single-process, no network); use backend='memory'"
        )
    raise ValueError(f"unknown backend {backend!r}")
