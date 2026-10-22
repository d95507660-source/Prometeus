"""Behaviour log: one JSONL line per decision of a :class:`~behavior.humanlike.HumanLikeWrapper`.

Separate file from the M1 decision log and the M4 signal log. Like ``agent/logging.py``
the module name shadows the stdlib only for relative imports; keep imports absolute.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from adapter.models import Action

__all__ = ["BehaviorLogger", "BehaviorRecord", "action_log"]

Clock = Callable[[], datetime]


def action_log(action: Action) -> dict[str, Any]:
    """``{"type", "amount"}`` of an action."""
    return {"type": action.type.value, "amount": action.amount}


@dataclass(frozen=True, slots=True)
class BehaviorRecord:
    """One decision as seen by the behaviour layer (everything but the timestamp)."""

    hand_id: str | None
    seat: int
    street: str | None
    base_action: Action
    final_action: Action
    error_applied: bool
    error_kind: str | None
    planned_delay_ms: float
    tilt_state: bool
    humanlike_level: float

    def as_log(self) -> dict[str, Any]:
        """JSON-friendly fields in log order."""
        return {
            "hand_id": self.hand_id,
            "seat": self.seat,
            "street": self.street,
            "base_action": action_log(self.base_action),
            "final_action": action_log(self.final_action),
            "error_applied": self.error_applied,
            "error_kind": self.error_kind,
            "planned_delay_ms": round(self.planned_delay_ms, 3),
            "tilt_state": self.tilt_state,
            "humanlike_level": self.humanlike_level,
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BehaviorLogger:
    """Appends ``{ts, **record}`` lines to ``path``; ``path=None`` is a no-op. Thread-safe, shareable by seats."""

    def __init__(self, path: Path | str | None, clock: Clock | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock or _utcnow
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: BehaviorRecord) -> None:
        """Write one record."""
        if self._fh is None:
            return
        line = json.dumps({"ts": self._clock().isoformat(), **record.as_log()}, separators=(",", ":"))
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def close(self) -> None:
        """Close the file. Idempotent."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> BehaviorLogger:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
