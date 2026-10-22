"""Data models shared by every :class:`~adapter.base.PokerAdapter` implementation.

All models are immutable dataclasses so that a snapshot handed to a bot can never
be mutated behind the engine's back. Chip amounts are plain ``int`` (sandbox chips,
no real money). :func:`to_jsonable` converts any model into JSON-friendly
primitives for the JSONL audit log.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Any

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


@dataclass(frozen=True, slots=True, order=True)
class Card:
    """A playing card. ``rank`` is 2..14 (14 = ace), ``suit`` is one of ``c d h s``."""

    rank: int
    suit: str

    def __post_init__(self) -> None:
        if not 2 <= self.rank <= 14:
            raise ValueError(f"invalid rank: {self.rank}")
        if self.suit not in SUIT_CHARS:
            raise ValueError(f"invalid suit: {self.suit!r}")

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse a two-character card such as ``"Ah"`` or ``"Tc"``."""
        if len(text) != 2 or text[0].upper() not in RANK_CHARS:
            raise ValueError(f"invalid card: {text!r}")
        return cls(RANK_CHARS.index(text[0].upper()) + 2, text[1].lower())

    def __str__(self) -> str:
        return f"{RANK_CHARS[self.rank - 2]}{self.suit}"


def cards(text: str) -> tuple[Card, ...]:
    """Parse a space-separated list of cards: ``cards("Ah Kd")``."""
    return tuple(Card.parse(t) for t in text.split())


class Street(str, enum.Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(str, enum.Enum):
    """Actions a player may send.

    For ``BET`` and ``RAISE`` the :attr:`Action.amount` is the *total* the player
    will have committed on the current street ("raise to"), not the increment.
    ``ALL_IN`` ignores ``amount`` and is normalised by the engine into a
    ``BET``, ``CALL`` or ``RAISE``.
    """

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class EventType(str, enum.Enum):
    SESSION_STARTED = "session_started"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    HAND_STARTED = "hand_started"
    DECK = "deck"
    HOLE_CARDS = "hole_cards"
    BLIND_POSTED = "blind_posted"
    TURN = "turn"
    ACTION = "action"
    ACTION_REJECTED = "action_rejected"
    STREET_DEALT = "street_dealt"
    SHOWDOWN = "showdown"
    POT_AWARDED = "pot_awarded"
    HAND_ENDED = "hand_ended"
    HAND_ABORTED = "hand_aborted"
    PLAYER_BUSTED = "player_busted"
    STACK_ADJUSTED = "stack_adjusted"
    TABLE_IDLE = "table_idle"


class Visibility(str, enum.Enum):
    """Who may see an event.

    * ``PUBLIC``  — every subscriber.
    * ``PRIVATE`` — only the subscriber bound to :attr:`Event.recipient_seat`.
    * ``AUDIT``   — never delivered to bots; written to the audit log only
      (full deck, seed, stack adjustments).
    """

    PUBLIC = "public"
    PRIVATE = "private"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True)
class Action:
    """An action submitted by the bot sitting in ``seat``.

    ``hand_id`` is optional; when given, the adapter rejects the action if it
    refers to a hand other than the one in progress (protects against stale
    decisions).
    """

    seat: int
    type: ActionType
    amount: int = 0
    hand_id: str | None = None

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("amount must be non-negative")


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Public view of one seat. ``hole_cards`` is ``None`` unless visible to the viewer."""

    seat: int
    player_id: str
    stack: int
    bet: int = 0
    committed: int = 0
    hole_cards: tuple[Card, ...] | None = None
    in_hand: bool = False
    has_folded: bool = False
    is_all_in: bool = False


@dataclass(frozen=True, slots=True)
class TableState:
    """Snapshot of the table from the point of view of one seat (or an observer).

    ``legal_actions``/``to_call``/``min_raise_to``/``max_raise_to`` describe the
    options of the seat in ``to_act_seat``; they are empty/zero when nobody is
    to act.
    """

    table_id: str
    max_seats: int
    small_blind: int
    big_blind: int
    players: tuple[PlayerState, ...]
    button_seat: int | None = None
    hand_id: str | None = None
    hand_number: int = 0
    street: Street | None = None
    board: tuple[Card, ...] = ()
    pot: int = 0
    current_bet: int = 0
    to_act_seat: int | None = None
    legal_actions: tuple[ActionType, ...] = ()
    to_call: int = 0
    min_raise_to: int = 0
    max_raise_to: int = 0

    def player(self, seat: int) -> PlayerState:
        for p in self.players:
            if p.seat == seat:
                return p
        raise KeyError(seat)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """A normalised, accepted action as stored in the hand history.

    ``amount`` is chips moved into the pot by this action; ``street_total`` is
    the player's total commitment on the street afterwards.
    """

    seat: int
    street: Street
    type: ActionType
    amount: int
    street_total: int
    is_all_in: bool
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class HandResult:
    """Complete history of a finished hand.

    ``revealed_cards`` contains hole cards shown at showdown plus the viewer's
    own cards; ``winnings`` is gross chips collected from pots, ``net`` is the
    stack delta over the hand.
    """

    hand_id: str
    hand_number: int
    table_id: str
    button_seat: int
    small_blind_seat: int
    big_blind_seat: int
    board: tuple[Card, ...]
    starting_stacks: Mapping[int, int]
    final_stacks: Mapping[int, int]
    blinds: Mapping[int, int]
    actions: tuple[ActionRecord, ...]
    pots: tuple[Pot, ...]
    winnings: Mapping[int, int]
    net: Mapping[int, int]
    revealed_cards: Mapping[int, tuple[Card, ...]]
    hand_ranks: Mapping[int, str]
    went_to_showdown: bool
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class Event:
    """Something that happened at the table. ``seq`` is strictly increasing per session."""

    type: EventType
    timestamp: datetime
    seq: int
    table_id: str
    hand_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    visibility: Visibility = Visibility.PUBLIC
    recipient_seat: int | None = None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert models, enums, cards and datetimes to JSON primitives."""
    if isinstance(obj, Card):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    return obj
