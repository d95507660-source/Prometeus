"""Append-only JSONL audit log: one event per line, flushed immediately.

The log contains *every* event, including ``AUDIT``-visibility records (seed,
full deck) that bots never see, so offline detectors can compare decisions
against hidden information.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from .models import Event, to_jsonable


class JsonlAuditLog:
    """Writes events to ``path``; with ``path=None`` it is a no-op sink."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None
        self._fh: IO[str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: Event) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(to_jsonable(event), ensure_ascii=False, separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def read(path: Path | str) -> Iterator[dict[str, Any]]:
        """Iterate over records of an existing log."""
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
