"""Card utilities and the 0..51 integer encoding used by the numeric core.

Card comes from adapter.models (a stdlib-only dataclass; adapter never imports pokermath, so no cycle).
Encoding: ``index = rank_idx * 4 + suit_idx`` with ``rank_idx = rank - 2`` (0..12) and suits ``c d h s``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from adapter.models import RANK_CHARS, SUIT_CHARS, Card

__all__ = [
    "Card",
    "Rank",
    "Suit",
    "canonical",
    "from_index",
    "full_deck_indices",
    "parse",
    "parse_many",
    "to_index",
]


class Rank(IntEnum):
    """Card rank with the same values as ``Card.rank`` (2..14)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def char(self) -> str:
        """One-letter symbol: ``2``..``9``, ``T``, ``J``, ``Q``, ``K``, ``A``."""
        return RANK_CHARS[self.value - 2]

    @classmethod
    def from_char(cls, char: str) -> Rank:
        """Parse a rank symbol (case-insensitive)."""
        pos = RANK_CHARS.find(char.upper())
        if pos < 0 or len(char) != 1:
            raise ValueError(f"invalid rank: {char!r}")
        return cls(pos + 2)


class Suit(IntEnum):
    """Suit index in the integer encoding (alphabetical: clubs, diamonds, hearts, spades)."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def char(self) -> str:
        """One-letter symbol used by ``Card.suit``."""
        return SUIT_CHARS[self.value]

    @classmethod
    def from_char(cls, char: str) -> Suit:
        """Parse a suit symbol (case-insensitive)."""
        pos = SUIT_CHARS.find(char.lower())
        if pos < 0 or len(char) != 1:
            raise ValueError(f"invalid suit: {char!r}")
        return cls(pos)


def parse(text: str) -> Card:
    """Parse ``"As"``, ``"Kd"``, ``"th"``… into a :class:`Card`."""
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"invalid card: {text!r}")
    return Card(Rank.from_char(text[0]).value, Suit.from_char(text[1]).char)


def parse_many(text: str) -> tuple[Card, ...]:
    """Parse cards separated by spaces/commas, or concatenated: ``"AsKd"``, ``"As Kd, 2c"``."""
    compact = "".join(text.replace(",", " ").split())
    if len(compact) % 2:
        raise ValueError(f"invalid card list: {text!r}")
    out = tuple(parse(compact[i : i + 2]) for i in range(0, len(compact), 2))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate cards in {text!r}")
    return out


def to_index(card: Card) -> int:
    """Card → 0..51."""
    return (card.rank - 2) * 4 + SUIT_CHARS.index(card.suit)


def from_index(index: int) -> Card:
    """0..51 → Card."""
    if not 0 <= index < 52:
        raise ValueError(f"card index out of range: {index}")
    return Card(index // 4 + 2, SUIT_CHARS[index % 4])


def canonical(cards: Iterable[Card]) -> tuple[Card, ...]:
    """Cards sorted strongest first (rank desc, then suit desc); duplicates rejected."""
    out = tuple(sorted(cards, key=to_index, reverse=True))
    if len(set(out)) != len(out):
        raise ValueError("duplicate cards")
    return out


def full_deck_indices() -> npt.NDArray[np.int64]:
    """All 52 indices as an int64 array."""
    return np.arange(52, dtype=np.int64)


def indices(cards: Iterable[Card]) -> list[int]:
    """Cards → list of indices (helper for the numeric core)."""
    return [to_index(c) for c in cards]
