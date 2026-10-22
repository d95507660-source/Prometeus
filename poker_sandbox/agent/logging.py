"""Decision log: one JSONL line per action an agent sent.

Kept in a separate file from the M0 audit log. The module name shadows the
stdlib only for relative imports: absolute imports are mandatory in this
package, do not rename.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO

from adapter.models import Action, TableState, to_jsonable


def state_hash(state: TableState, seat: int) -> str:
    """sha256 of the canonical JSON of ``state`` with every other seat's hole cards removed."""
    masked = replace(
        state,
        players=tuple(p if p.seat == seat else replace(p, hole_cards=None) for p in state.players),
    )
    canonical = json.dumps(to_jsonable(masked), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class DecisionLogger:
    """Appends decision records to ``path``; ``path=None`` is a no-op sink."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None
        self._fh: IO[str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(
        self,
        *,
        state: TableState,
        seat: int,
        action: Action,
        latency_ms: float,
        strategy_name: str,
        reason: str,
    ) -> None:
        """Write one record: {ts, hand_id, seat, state_hash, action, latency_ms, strategy_name, reason}."""
        if self._fh is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "hand_id": state.hand_id,
            "seat": seat,
            "state_hash": state_hash(state, seat),
            "action": {"type": action.type.value, "amount": action.amount},
            "latency_ms": round(latency_ms, 3),
            "strategy_name": strategy_name,
            "reason": reason,
        }
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file. Idempotent."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> DecisionLogger:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
