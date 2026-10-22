"""Agent counters. Mutated only from the event-loop thread."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMetrics:
    """Decision-loop counters.

    ``decisions_total`` counts actions actually SENT and accepted, fallbacks
    included. A send rejected with ``IllegalActionError`` is not counted (it goes
    to ``errors_total``); retries are not counted either (``retries_total``).
    """

    decisions_total: int = 0
    decisions_by_action: Counter[str] = field(default_factory=Counter)
    decision_latency_ms: list[float] = field(default_factory=list)
    timeouts_total: int = 0
    errors_total: int = 0
    retries_total: int = 0
    events_dropped_total: int = 0
    resyncs_total: int = 0
    foreign_events_total: int = 0

    def record_decision(self, action_type: str, latency_ms: float) -> None:
        """Count one accepted action."""
        self.decisions_total += 1
        self.decisions_by_action[action_type] += 1
        self.decision_latency_ms.append(latency_ms)

    @property
    def latency_mean_ms(self) -> float:
        """Mean decision latency, 0 when empty."""
        lat = self.decision_latency_ms
        return sum(lat) / len(lat) if lat else 0.0

    def latency_percentile_ms(self, q: float) -> float:
        """Nearest-rank percentile, ``q`` in (0, 100]; 0 when empty."""
        if not 0 < q <= 100:
            raise ValueError("q must be in (0, 100]")
        lat = sorted(self.decision_latency_ms)
        if not lat:
            return 0.0
        return lat[max(0, math.ceil(q / 100 * len(lat)) - 1)]

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict copy for printing/serialisation."""
        return {
            "decisions_total": self.decisions_total,
            "decisions_by_action": dict(sorted(self.decisions_by_action.items())),
            "latency_ms": {
                "mean": round(self.latency_mean_ms, 3),
                "p50": round(self.latency_percentile_ms(50), 3),
                "p95": round(self.latency_percentile_ms(95), 3),
            },
            "timeouts_total": self.timeouts_total,
            "errors_total": self.errors_total,
            "retries_total": self.retries_total,
            "events_dropped_total": self.events_dropped_total,
            "resyncs_total": self.resyncs_total,
            "foreign_events_total": self.foreign_events_total,
        }

    def merge(self, other: AgentMetrics) -> AgentMetrics:
        """Sum of two metric sets (for table-wide reports)."""
        out = AgentMetrics(
            decisions_total=self.decisions_total + other.decisions_total,
            decisions_by_action=self.decisions_by_action + other.decisions_by_action,
            decision_latency_ms=self.decision_latency_ms + other.decision_latency_ms,
        )
        for name in ("timeouts_total", "errors_total", "retries_total", "events_dropped_total",
                     "resyncs_total", "foreign_events_total"):
            setattr(out, name, getattr(self, name) + getattr(other, name))
        return out
