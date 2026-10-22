"""Adapter configuration (pydantic v2).

The same :class:`AdapterConfig` is passed to every adapter. Network adapters use
``base_url``/``api_token``/``limits``; :class:`~adapter.mock.MockAdapter` ignores
the network fields and uses the table/seed settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, SecretStr, model_validator


class Stakes(BaseModel):
    """Blinds and starting stack, in sandbox chips."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    small_blind: int = Field(default=1, ge=1)
    big_blind: int = Field(default=2, ge=1)
    starting_stack: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def _check_blinds(self) -> Stakes:
        if self.big_blind < self.small_blind:
            raise ValueError("big_blind must be >= small_blind")
        return self


class RateLimits(BaseModel):
    """Client-side limits.

    ``max_actions_per_second`` and ``action_timeout_s`` are meant for network
    adapters; ``max_hands`` is honoured by every adapter (including the mock)
    and stops dealing once reached.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_actions_per_second: float | None = Field(default=None, gt=0)
    action_timeout_s: float = Field(default=30.0, gt=0)
    max_hands: int | None = Field(default=None, ge=1)


class AdapterConfig(BaseModel):
    """Connection and table settings.

    Attributes:
        base_url: Endpoint of the sandbox API (unused by the mock).
        api_token: Auth token; stored as ``SecretStr`` so it never leaks into logs.
        table_id: Identifier of the table, also used as hand-id prefix.
        seats: Number of seats, 2 (heads-up) to 6 (6-max).
        hero_seat: Seat whose private information this adapter instance sees
            (hole cards in :meth:`get_table_state`, private events). ``None``
            means an observer that sees only public information.
        player_ids: Optional names per seat; defaults to ``bot_0..bot_{n-1}``.
        seed: RNG seed for reproducible deals. ``None`` picks a random seed and
            records it in the audit log so the run can still be replayed.
        log_path: JSONL audit log destination; ``None`` disables file logging.
        auto_deal: Start the next hand automatically when the previous ends.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: AnyUrl | None = None
    api_token: SecretStr | None = None
    table_id: str = Field(default="sandbox-1", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    seats: int = Field(default=6, ge=2, le=6)
    hero_seat: int | None = 0
    player_ids: tuple[str, ...] | None = None
    stakes: Stakes = Field(default_factory=Stakes)
    limits: RateLimits = Field(default_factory=RateLimits)
    seed: int | None = None
    log_path: Path | None = None
    auto_deal: bool = True

    @model_validator(mode="after")
    def _check_seats(self) -> AdapterConfig:
        if self.hero_seat is not None and not 0 <= self.hero_seat < self.seats:
            raise ValueError(f"hero_seat must be in [0, {self.seats})")
        if self.player_ids is not None:
            if len(self.player_ids) != self.seats:
                raise ValueError("player_ids must have exactly `seats` entries")
            if len(set(self.player_ids)) != len(self.player_ids):
                raise ValueError("player_ids must be unique")
        return self

    def resolved_player_ids(self) -> tuple[str, ...]:
        return self.player_ids or tuple(f"bot_{i}" for i in range(self.seats))

    def redacted(self) -> dict[str, Any]:
        """JSON-safe dump with secrets masked, suitable for the audit log."""
        return self.model_dump(mode="json")
