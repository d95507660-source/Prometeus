"""Signal protocol between coordinated bots: coarse hand strength and intended line only.

Messages never carry cards: there are no card fields, ``hand_id`` is validated,
and every encoded payload is re-checked for card-like tokens before it leaves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any, Final

from adapter.models import Card
from pokermath.equity import equity_vs_range
from pokermath.ranges import Range

__all__ = [
    "AVERAGE_RANGE",
    "CARD_TOKEN",
    "Bucket",
    "Line",
    "Message",
    "SignalIntent",
    "SignalStrength",
    "bucket_for_equity",
    "decode",
    "encode",
    "strength_signal",
]

PROTOCOL_VERSION: Final = 1
CARD_TOKEN: Final = re.compile(r"(?<![A-Za-z0-9])[2-9TJQKAtjqka][cdhsCDHS](?![A-Za-z0-9])")
FORBIDDEN_KEYS: Final = frozenset({"hole_cards", "hole", "cards", "card", "board", "hand"})
_HAND_ID: Final = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Reference "average" range the strength bucket is measured against (~35% of hands).
AVERAGE_RANGE: Final = "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 98s, 87s, A7o+, KTo+, QTo+, JTo"


class Bucket(StrEnum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    MONSTER = "monster"

    @property
    def level(self) -> int:
        """0 (weak) .. 3 (monster)."""
        return list(Bucket).index(self)


class Line(StrEnum):
    I_PRESSURE = "I_pressure"
    I_YIELD = "I_yield"
    I_CALL_DOWN = "I_call_down"
    I_FOLD_TO_YOU = "I_fold_to_you"


def _check_common(seat: int, hand_id: str) -> None:
    if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat < 10:
        raise ValueError(f"invalid seat {seat!r}")
    if not isinstance(hand_id, str) or not _HAND_ID.match(hand_id) or CARD_TOKEN.search(hand_id):
        raise ValueError(f"invalid hand_id {hand_id!r}")  # also blocks smuggling cards via the id


@dataclass(frozen=True, slots=True)
class SignalStrength:
    """Discretised strength of the sender's hand in ``hand_id``."""

    seat: int
    hand_id: str
    bucket: Bucket

    def __post_init__(self) -> None:
        _check_common(self.seat, self.hand_id)
        object.__setattr__(self, "bucket", Bucket(self.bucket))


@dataclass(frozen=True, slots=True)
class SignalIntent:
    """Line the sender proposes to take in ``hand_id``."""

    seat: int
    hand_id: str
    line: Line

    def __post_init__(self) -> None:
        _check_common(self.seat, self.hand_id)
        object.__setattr__(self, "line", Line(self.line))


Message = SignalStrength | SignalIntent


def _payload(msg: Message) -> dict[str, Any]:
    if isinstance(msg, SignalStrength):
        return {"v": PROTOCOL_VERSION, "kind": "strength", "seat": msg.seat, "hand_id": msg.hand_id,
                "bucket": msg.bucket.value}
    return {"v": PROTOCOL_VERSION, "kind": "intent", "seat": msg.seat, "hand_id": msg.hand_id,
            "line": msg.line.value}


def _assert_card_free(data: dict[str, Any], text: str) -> None:
    if FORBIDDEN_KEYS & data.keys():
        raise ValueError(f"forbidden keys in signal: {sorted(FORBIDDEN_KEYS & data.keys())}")
    if CARD_TOKEN.search(text):
        raise ValueError("signal payload contains a card-like token")


def encode(msg: Message) -> bytes:
    """Canonical JSON bytes; raises if the result could reveal a card."""
    data = _payload(msg)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    _assert_card_free(data, text)
    return text.encode()


def decode(raw: bytes) -> Message:
    """Strict inverse of :func:`encode` (unknown or forbidden keys are rejected)."""
    text = raw.decode()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("signal must be a JSON object")
    _assert_card_free(data, text)
    kind = data.get("kind")
    expected = {"v", "kind", "seat", "hand_id", "bucket" if kind == "strength" else "line"}
    if set(data) != expected or data["v"] != PROTOCOL_VERSION:
        raise ValueError(f"unexpected signal fields: {sorted(data)}")
    if kind == "strength":
        return SignalStrength(data["seat"], data["hand_id"], Bucket(data["bucket"]))
    if kind == "intent":
        return SignalIntent(data["seat"], data["hand_id"], Line(data["line"]))
    raise ValueError(f"unknown signal kind {kind!r}")


def bucket_for_equity(equity: float) -> Bucket:
    """weak < 0.40 ≤ medium < 0.60 ≤ strong < 0.80 ≤ monster."""
    if equity < 0.40:
        return Bucket.WEAK
    if equity < 0.60:
        return Bucket.MEDIUM
    if equity < 0.80:
        return Bucket.STRONG
    return Bucket.MONSTER


@cache
def _average_range() -> Range:
    return Range.parse(AVERAGE_RANGE)


def strength_signal(
    seat: int, hand_id: str, hole: tuple[Card, ...], board: tuple[Card, ...], samples: int, seed: int
) -> SignalStrength:
    """Bucket the sender's equity vs :data:`AVERAGE_RANGE`; only the bucket leaves this function."""
    try:
        eq = equity_vs_range(hole, board, _average_range(), samples, seed)
    except ValueError:  # average range fully blocked: treat as medium
        eq = 0.5
    return SignalStrength(seat, hand_id, bucket_for_equity(eq))
