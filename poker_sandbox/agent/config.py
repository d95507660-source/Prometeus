"""Agent configuration (pydantic v2)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    """Tuning of the decision loop.

    Attributes:
        queue_maxsize: Capacity of the event queue; on overflow the oldest event is dropped.
        decision_timeout_s: Budget for ``Strategy.decide``; on expiry a passive fallback is sent.
        retry_base_s: First backoff delay for transient adapter errors.
        retry_cap_s: Upper bound of a single backoff delay.
        retry_max: Maximum number of retries after the first attempt (0 disables retries).
        decision_log_path: JSONL decision log; ``None`` disables it. Never the M0 audit log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    queue_maxsize: int = Field(default=1024, ge=1)
    decision_timeout_s: float = Field(default=5.0, gt=0)
    retry_base_s: float = Field(default=0.05, ge=0)
    retry_cap_s: float = Field(default=2.0, ge=0)
    retry_max: int = Field(default=3, ge=0)
    decision_log_path: Path | None = None
