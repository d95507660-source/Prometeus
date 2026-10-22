"""Offline No-Limit Texas Hold'em engine behind the :class:`PokerAdapter` contract.

No network access is performed. The table is driven locally: any seat may send
actions (the local harness plays every bot), while information is still
filtered per seat for :meth:`MockAdapter.view` and event subscribers.

Rules implemented:

* heads-up (button posts the small blind and acts first preflop, last
  postflop) and 3-6 handed play;
* no-limit betting: minimum bet = big blind, minimum raise = size of the last
  full raise, an all-in for less than a full raise does not reopen betting for
  players who have already acted;
* side pots, uncalled-bet return, split pots (odd chips go to the first winner
  left of the button);
* all-in run-outs and showdown with 7-card evaluation.

Simplifications (documented, intentional): no antes, no straddles, no dead
button (the button moves to the next seat with chips), no burn cards, all
showdown hands are tabled (no mucking).
"""

from __future__ import annotations

import itertools
import logging
import random
import secrets
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .audit import JsonlAuditLog
from .base import (
    EventCallback,
    HandNotFoundError,
    IllegalActionError,
    NotConnectedError,
    PokerAdapter,
    TableClosedError,
    Unsubscribe,
)
from .config import AdapterConfig
from .models import (
    SUIT_CHARS,
    Action,
    ActionRecord,
    ActionType,
    Card,
    Event,
    EventType,
    HandResult,
    PlayerState,
    Pot,
    Street,
    TableState,
    Visibility,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Hand evaluation
# --------------------------------------------------------------------------- #

HAND_CATEGORIES = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)


@dataclass(frozen=True, slots=True, order=True)
class HandRank:
    """Comparable strength of a 5-card hand: higher compares greater."""

    category: int
    tiebreak: tuple[int, ...]

    @property
    def name(self) -> str:
        return HAND_CATEGORIES[self.category]


def _rank_five(five: Sequence[Card]) -> HandRank:
    ranks = sorted((c.rank for c in five), reverse=True)
    is_flush = len({c.suit for c in five}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_high: int | None = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5  # wheel
    if straight_high is not None and is_flush:
        return HandRank(8, (straight_high,))

    # Group by (count, rank) so e.g. trips sort before the pair in a full house.
    groups = sorted(Counter(ranks).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [count for _, count in groups]
    by_group = tuple(rank for rank, _ in groups)
    if shape == [4, 1]:
        return HandRank(7, by_group)
    if shape == [3, 2]:
        return HandRank(6, by_group)
    if is_flush:
        return HandRank(5, tuple(ranks))
    if straight_high is not None:
        return HandRank(4, (straight_high,))
    if shape == [3, 1, 1]:
        return HandRank(3, by_group)
    if shape == [2, 2, 1]:
        return HandRank(2, by_group)
    if shape == [2, 1, 1, 1]:
        return HandRank(1, by_group)
    return HandRank(0, tuple(ranks))


def evaluate_hand(hand: Sequence[Card]) -> HandRank:
    """Best 5-card rank out of 5-7 cards."""
    if not 5 <= len(hand) <= 7:
        raise ValueError("evaluate_hand needs 5 to 7 cards")
    if len(set(hand)) != len(hand):
        raise ValueError("duplicate cards")
    return max(_rank_five(combo) for combo in itertools.combinations(hand, 5))


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in SUIT_CHARS for rank in range(2, 15)]


# --------------------------------------------------------------------------- #
# Internal mutable state
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Seat:
    seat: int
    player_id: str
    stack: int
    hole: tuple[Card, ...] = ()
    bet: int = 0
    committed: int = 0
    in_hand: bool = False
    folded: bool = False
    all_in: bool = False

    @property
    def active(self) -> bool:
        """Still able to make decisions this hand."""
        return self.in_hand and not self.folded and not self.all_in

    @property
    def contending(self) -> bool:
        return self.in_hand and not self.folded


@dataclass(slots=True)
class _Hand:
    hand_id: str
    number: int
    button: int
    sb_seat: int
    bb_seat: int
    board_cards: tuple[Card, ...]
    starting_stacks: dict[int, int]
    started_at: datetime
    street: Street = Street.PREFLOP
    board_shown: int = 0
    current_bet: int = 0
    last_raise_size: int = 0
    pending: set[int] = field(default_factory=set)
    may_raise: set[int] = field(default_factory=set)
    current: int | None = None
    blinds: dict[int, int] = field(default_factory=dict)
    actions: list[ActionRecord] = field(default_factory=list)

    @property
    def board(self) -> tuple[Card, ...]:
        return self.board_cards[: self.board_shown]


_NEXT_STREET = {Street.PREFLOP: (Street.FLOP, 3), Street.FLOP: (Street.TURN, 4), Street.TURN: (Street.RIVER, 5)}


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class MockAdapter(PokerAdapter):
    """Deterministic in-process poker table.

    Args:
        clock: Source of event timestamps (UTC). Inject a fake for fully
            reproducible logs.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._config: AdapterConfig | None = None
        self._connected = False
        self._seed: int | None = None
        self._rng = random.Random()
        self._audit = JsonlAuditLog(None)
        self._seats: dict[int, _Seat] = {}
        self._hand: _Hand | None = None
        self._button: int | None = None
        self._hand_counter = 0
        self._history: dict[str, HandResult] = {}
        self._subscribers: dict[int, tuple[EventCallback, int | None]] = {}
        self._next_sub_id = 0
        self._queue: deque[Event] = deque()
        self._dispatching = False
        self._seq = 0

    # ----------------------------------------------------------- contract --

    def connect(self, config: AdapterConfig) -> None:
        if self._connected:
            raise RuntimeError("already connected")
        self._config = config
        self._seed = config.seed if config.seed is not None else secrets.randbits(63)
        self._rng = random.Random(self._seed)
        self._audit = JsonlAuditLog(config.log_path)
        self._seats = {
            i: _Seat(i, pid, config.stakes.starting_stack) for i, pid in enumerate(config.resolved_player_ids())
        }
        self._hand = None
        self._button = None
        self._hand_counter = 0
        self._history.clear()
        self._seq = 0
        self._connected = True
        self._emit(
            EventType.SESSION_STARTED,
            {"seed": self._seed, "config": config.redacted(), "engine": "mock-nlhe"},
            visibility=Visibility.AUDIT,
        )
        self._emit(
            EventType.CONNECTED,
            {"seats": config.seats, "players": {s.seat: s.player_id for s in self._seats.values()}},
        )
        if config.auto_deal:
            self._maybe_auto_deal()
        self._drain()

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._hand is not None:
            self._emit(EventType.HAND_ABORTED, {"reason": "disconnect"}, hand_id=self._hand.hand_id)
            self._hand = None
        self._emit(EventType.DISCONNECTED, {"hands_played": self._hand_counter})
        self._drain()
        self._connected = False
        self._audit.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_table_state(self) -> TableState:
        return self.view(self._cfg.hero_seat)

    def send_action(self, action: Action) -> None:
        self._require_connected()
        try:
            self._apply(action)
        except IllegalActionError as exc:
            self._emit(
                EventType.ACTION_REJECTED,
                {"seat": action.seat, "type": action.type, "amount": action.amount,
                 "sent_hand_id": action.hand_id, "reason": str(exc)},
                hand_id=self._hand.hand_id if self._hand else None,
            )
            self._drain()
            raise
        self._drain()

    def on_event(self, callback: EventCallback, *, seat: int | None = None) -> Unsubscribe:
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self._subscribers[sub_id] = (callback, seat)

        def unsubscribe() -> None:
            self._subscribers.pop(sub_id, None)

        return unsubscribe

    def get_hand_history(self, hand_id: str) -> HandResult:
        try:
            return self._history[hand_id]
        except KeyError:
            raise HandNotFoundError(hand_id) from None

    # --------------------------------------------------- mock-only extras --

    @property
    def seed(self) -> int | None:
        """Seed actually used for this session (also written to the audit log)."""
        return self._seed

    @property
    def hand_ids(self) -> tuple[str, ...]:
        return tuple(self._history)

    def view(self, seat: int | None) -> TableState:
        """Table snapshot as seen by ``seat`` (``None`` = public observer)."""
        self._require_connected()
        cfg = self._cfg
        h = self._hand
        players = tuple(
            PlayerState(
                seat=s.seat,
                player_id=s.player_id,
                stack=s.stack,
                bet=s.bet if h else 0,
                committed=s.committed if h else 0,
                hole_cards=s.hole if (h and s.in_hand and s.seat == seat) else None,
                in_hand=bool(h) and s.in_hand,
                has_folded=bool(h) and s.folded,
                is_all_in=bool(h) and s.all_in,
            )
            for s in self._seats.values()
        )
        base: dict[str, Any] = {
            "table_id": cfg.table_id,
            "max_seats": cfg.seats,
            "small_blind": cfg.stakes.small_blind,
            "big_blind": cfg.stakes.big_blind,
            "players": players,
            "button_seat": self._button,
            "hand_number": self._hand_counter,
        }
        if h is None:
            return TableState(**base)
        legal, to_call, min_to, max_to = self._legal(h.current) if h.current is not None else ((), 0, 0, 0)
        return TableState(
            **base,
            hand_id=h.hand_id,
            street=h.street,
            board=h.board,
            pot=sum(s.committed for s in self._seats.values()),
            current_bet=h.current_bet,
            to_act_seat=h.current,
            legal_actions=legal,
            to_call=to_call,
            min_raise_to=min_to,
            max_raise_to=max_to,
        )

    def start_hand(
        self,
        *,
        hole_cards: Mapping[int, Sequence[Card]] | None = None,
        board: Sequence[Card] | None = None,
        button: int | None = None,
    ) -> str:
        """Deal a new hand manually (use with ``auto_deal=False``).

        ``hole_cards``/``board`` stack the deck for tests: the given cards are
        dealt to the given seats / board, the rest comes from the seeded
        shuffle. ``button`` places the dealer button explicitly. Returns the
        new ``hand_id``.
        """
        self._require_connected()
        if self._hand is not None:
            raise TableClosedError("a hand is already in progress")
        reason = self._cannot_deal_reason()
        if reason:
            raise TableClosedError(reason)
        if button is not None and self._seats[button].stack == 0:
            raise ValueError(f"seat {button} has no chips and cannot hold the button")
        hand_id = self._start_hand(hole_cards or {}, tuple(board or ()), button)
        self._drain()
        return hand_id

    def set_stack(self, seat: int, chips: int) -> None:
        """Adjust a stack between hands (test setup). Logged as an audit event."""
        self._require_connected()
        if self._hand is not None:
            raise RuntimeError("cannot change stacks during a hand")
        if chips < 0:
            raise ValueError("chips must be >= 0")
        old = self._seats[seat].stack
        self._seats[seat].stack = chips
        self._emit(EventType.STACK_ADJUSTED, {"seat": seat, "from": old, "to": chips}, visibility=Visibility.AUDIT)
        self._drain()

    # ------------------------------------------------------------ helpers --

    @property
    def _cfg(self) -> AdapterConfig:
        if self._config is None:
            raise NotConnectedError("adapter is not connected")
        return self._config

    def _require_connected(self) -> None:
        if not self._connected:
            raise NotConnectedError("adapter is not connected")

    def _emit(
        self,
        etype: EventType,
        payload: Mapping[str, Any],
        *,
        hand_id: str | None = None,
        visibility: Visibility = Visibility.PUBLIC,
        recipient_seat: int | None = None,
    ) -> None:
        """Log immediately, deliver later (from :meth:`_drain`, once state is consistent)."""
        self._seq += 1
        event = Event(
            type=etype,
            timestamp=self._clock(),
            seq=self._seq,
            table_id=self._cfg.table_id,
            hand_id=hand_id,
            payload=dict(payload),
            visibility=visibility,
            recipient_seat=recipient_seat,
        )
        self._audit.write(event)
        if visibility is not Visibility.AUDIT:
            self._queue.append(event)

    def _drain(self) -> None:
        if self._dispatching:
            return  # re-entrant call from a callback; the outer loop delivers
        self._dispatching = True
        try:
            while self._queue:
                event = self._queue.popleft()
                for callback, seat in list(self._subscribers.values()):
                    viewer = seat if seat is not None else self._cfg.hero_seat
                    if event.visibility is Visibility.PRIVATE and event.recipient_seat != viewer:
                        continue
                    try:
                        callback(event)
                    except Exception:
                        logger.exception("event callback failed for %s", event.type.value)
        finally:
            self._dispatching = False

    def _clockwise(self, after: int, candidates: Iterable[int]) -> list[int]:
        """``candidates`` ordered clockwise starting from the seat after ``after``."""
        pool = set(candidates)
        n = self._cfg.seats
        return [s for s in ((after + i) % n for i in range(1, n + 1)) if s in pool]

    def _next_pending(self, after: int) -> int:
        h = self._hand
        assert h is not None
        return self._clockwise(after, h.pending)[0]

    def _cannot_deal_reason(self) -> str | None:
        max_hands = self._cfg.limits.max_hands
        if max_hands is not None and self._hand_counter >= max_hands:
            return f"max_hands={max_hands} reached"
        if sum(1 for s in self._seats.values() if s.stack > 0) < 2:
            return "fewer than two players with chips"
        return None

    def _maybe_auto_deal(self) -> None:
        reason = self._cannot_deal_reason()
        if reason:
            self._emit(EventType.TABLE_IDLE, {"reason": reason})
        else:
            self._start_hand({}, (), None)

    # ---------------------------------------------------------- hand flow --

    def _start_hand(
        self,
        preset_hole: Mapping[int, Sequence[Card]],
        preset_board: tuple[Card, ...],
        button: int | None,
    ) -> str:
        cfg = self._cfg
        eligible = sorted(s.seat for s in self._seats.values() if s.stack > 0)
        if button is None:
            if self._button is None:
                button = self._rng.choice(eligible)
            else:
                button = self._clockwise(self._button, eligible)[0]
        self._button = button

        # The shuffle always consumes the RNG the same way, so presets never
        # shift the random stream of later hands.
        deck = full_deck()
        self._rng.shuffle(deck)
        reserved = [c for cs in preset_hole.values() for c in cs] + list(preset_board)
        if len(set(reserved)) != len(reserved):
            raise ValueError("duplicate cards in preset")
        if len(preset_board) > 5 or any(len(cs) > 2 for cs in preset_hole.values()):
            raise ValueError("preset too long")
        if set(preset_hole) - set(eligible):
            raise ValueError("preset hole cards for a seat not in the hand")
        deck = [c for c in deck if c not in set(reserved)]

        deal_order = self._clockwise(button, eligible)
        holes: dict[int, list[Card]] = {s: list(preset_hole.get(s, ())) for s in deal_order}
        for _ in range(2):
            for s in deal_order:
                if len(holes[s]) < 2:
                    holes[s].append(deck.pop())
        board_cards = tuple(preset_board) + tuple(deck.pop() for _ in range(5 - len(preset_board)))

        if len(eligible) == 2:
            sb_seat, bb_seat = button, deal_order[0]
        else:
            sb_seat, bb_seat = deal_order[0], deal_order[1]

        self._hand_counter += 1
        hand_id = f"{cfg.table_id}-{self._hand_counter:06d}"
        for p in self._seats.values():
            p.bet = p.committed = 0
            p.folded = p.all_in = False
            p.in_hand = p.seat in holes
            p.hole = tuple(holes.get(p.seat, ()))
        h = _Hand(
            hand_id=hand_id,
            number=self._hand_counter,
            button=button,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            board_cards=board_cards,
            starting_stacks={s: self._seats[s].stack for s in eligible},
            started_at=self._clock(),
        )
        self._hand = h

        self._emit(
            EventType.HAND_STARTED,
            {"hand_number": h.number, "button": button, "small_blind_seat": sb_seat,
             "big_blind_seat": bb_seat, "stacks": dict(h.starting_stacks)},
            hand_id=hand_id,
        )
        self._emit(
            EventType.DECK,
            {"seed": self._seed, "hole_cards": {s: list(c) for s, c in holes.items()}, "board": list(board_cards)},
            hand_id=hand_id,
            visibility=Visibility.AUDIT,
        )
        for s in deal_order:
            self._emit(
                EventType.HOLE_CARDS, {"seat": s, "cards": list(holes[s])},
                hand_id=hand_id, visibility=Visibility.PRIVATE, recipient_seat=s,
            )

        self._post_blind(sb_seat, cfg.stakes.small_blind, "small")
        self._post_blind(bb_seat, cfg.stakes.big_blind, "big")
        h.current_bet = max(cfg.stakes.big_blind, max(self._seats[s].bet for s in eligible))
        h.last_raise_size = cfg.stakes.big_blind
        h.pending = {s for s in eligible if self._seats[s].active}
        h.may_raise = set(h.pending)
        self._progress(after=bb_seat)
        return hand_id

    def _post_blind(self, seat: int, amount: int, kind: str) -> None:
        h = self._hand
        assert h is not None
        paid = self._commit(self._seats[seat], amount)
        h.blinds[seat] = paid
        self._emit(
            EventType.BLIND_POSTED,
            {"seat": seat, "blind": kind, "amount": paid, "is_all_in": self._seats[seat].all_in},
            hand_id=h.hand_id,
        )

    def _commit(self, p: _Seat, chips: int) -> int:
        paid = min(chips, p.stack)
        p.stack -= paid
        p.bet += paid
        p.committed += paid
        if p.stack == 0:
            p.all_in = True
        return paid

    def _legal(self, seat: int) -> tuple[tuple[ActionType, ...], int, int, int]:
        """Legal actions, to_call, min_raise_to and max_raise_to for ``seat``."""
        h = self._hand
        assert h is not None
        p = self._seats[seat]
        to_call = h.current_bet - p.bet
        max_to = p.bet + p.stack
        min_to = min(h.current_bet + h.last_raise_size, max_to)
        can_raise = seat in h.may_raise and max_to > h.current_bet
        legal: list[ActionType] = []
        if to_call > 0:
            legal += [ActionType.FOLD, ActionType.CALL]
        else:
            legal.append(ActionType.CHECK)
        if can_raise:
            legal.append(ActionType.BET if h.current_bet == 0 else ActionType.RAISE)
        if can_raise or 0 < p.stack <= to_call:
            legal.append(ActionType.ALL_IN)
        return tuple(legal), min(to_call, p.stack), (min_to if can_raise else 0), (max_to if can_raise else 0)

    def _apply(self, action: Action) -> None:
        h = self._hand
        if h is None:
            raise IllegalActionError("no hand in progress")
        if action.hand_id is not None and action.hand_id != h.hand_id:
            raise IllegalActionError(f"stale action for hand {action.hand_id}, current is {h.hand_id}")
        if action.seat != h.current:
            raise IllegalActionError(f"not seat {action.seat}'s turn (to act: {h.current})")
        p = self._seats[action.seat]
        to_call = h.current_bet - p.bet
        max_to = p.bet + p.stack
        kind, amount = action.type, action.amount
        if kind is ActionType.ALL_IN:
            if h.current_bet == 0:
                kind, amount = ActionType.BET, max_to
            elif max_to <= h.current_bet:
                kind = ActionType.CALL
            else:
                kind, amount = ActionType.RAISE, max_to

        before = p.committed
        if kind is ActionType.FOLD:
            if to_call == 0:
                raise IllegalActionError("cannot fold when checking is free; use CHECK")
            p.folded = True
        elif kind is ActionType.CHECK:
            if to_call != 0:
                raise IllegalActionError(f"cannot check facing a bet of {to_call}")
        elif kind is ActionType.CALL:
            if to_call == 0:
                raise IllegalActionError("nothing to call; use CHECK")
            self._commit(p, to_call)
        elif kind in (ActionType.BET, ActionType.RAISE):
            self._bet_or_raise(p, kind, amount, max_to)
        else:  # pragma: no cover - exhaustive enum
            raise IllegalActionError(f"unknown action {kind}")

        h.pending.discard(p.seat)
        h.may_raise.discard(p.seat)
        record = ActionRecord(
            seat=p.seat, street=h.street, type=kind, amount=p.committed - before,
            street_total=p.bet, is_all_in=p.all_in, timestamp=self._clock(),
        )
        h.actions.append(record)
        self._emit(
            EventType.ACTION,
            {"seat": p.seat, "street": h.street, "type": kind, "amount": record.amount,
             "street_total": p.bet, "is_all_in": p.all_in, "requested": action.type,
             "stack": p.stack, "pot": sum(s.committed for s in self._seats.values())},
            hand_id=h.hand_id,
        )
        self._progress(after=p.seat)

    def _bet_or_raise(self, p: _Seat, kind: ActionType, amount: int, max_to: int) -> None:
        h = self._hand
        assert h is not None
        if kind is ActionType.BET and h.current_bet != 0:
            raise IllegalActionError(f"there is already a bet of {h.current_bet}; use RAISE")
        if kind is ActionType.RAISE and h.current_bet == 0:
            raise IllegalActionError("nothing to raise; use BET")
        if p.seat not in h.may_raise:
            raise IllegalActionError("betting was not reopened for this seat; only CALL or FOLD")
        if amount > max_to:
            raise IllegalActionError(f"{kind.value} to {amount} exceeds stack (max {max_to})")
        if amount <= h.current_bet:
            raise IllegalActionError(f"{kind.value} to {amount} must exceed current bet {h.current_bet}")
        min_to = h.current_bet + h.last_raise_size
        if amount < min_to and amount != max_to:
            raise IllegalActionError(f"{kind.value} to {amount} below minimum {min_to} (only allowed all-in)")

        raise_size = amount - h.current_bet
        full = raise_size >= h.last_raise_size
        self._commit(p, amount - p.bet)
        h.current_bet = amount
        others = {s.seat for s in self._seats.values() if s.active and s.seat != p.seat}
        h.pending = set(others)
        if full:
            h.last_raise_size = raise_size
        if full or kind is ActionType.BET:
            # An opening bet always opens the action, even an all-in below the BB.
            h.may_raise = set(others)

    def _progress(self, after: int) -> None:
        """Advance after an action: next player, next street, or end of hand."""
        h = self._hand
        assert h is not None
        contenders = [s for s in self._seats.values() if s.contending]
        if len(contenders) == 1:
            self._finish_hand()
            return
        active = [s for s in contenders if not s.all_in]
        if len(active) == 1 and active[0].bet >= h.current_bet:
            h.pending.clear()  # nobody left to bet against
        if h.pending:
            self._prompt(self._next_pending(after))
            return

        while True:
            for s in self._seats.values():
                s.bet = 0
            h.current_bet = 0
            h.last_raise_size = self._cfg.stakes.big_blind
            h.current = None
            if h.street is Street.RIVER:
                self._finish_hand()
                return
            h.street, h.board_shown = _NEXT_STREET[h.street]
            new_cards = h.board_cards[{Street.FLOP: 0, Street.TURN: 3, Street.RIVER: 4}[h.street]: h.board_shown]
            self._emit(
                EventType.STREET_DEALT,
                {"street": h.street, "cards": list(new_cards), "board": list(h.board)},
                hand_id=h.hand_id,
            )
            if len(active) >= 2:
                h.pending = {s.seat for s in active}
                h.may_raise = set(h.pending)
                self._prompt(self._next_pending(h.button))
                return
            # Everyone (or all but one) is all-in: run the board out.

    def _prompt(self, seat: int) -> None:
        h = self._hand
        assert h is not None
        h.current = seat
        legal, to_call, min_to, max_to = self._legal(seat)
        self._emit(
            EventType.TURN,
            {"seat": seat, "street": h.street, "legal_actions": list(legal), "to_call": to_call,
             "min_raise_to": min_to, "max_raise_to": max_to, "current_bet": h.current_bet},
            hand_id=h.hand_id,
        )

    def _build_pots(self) -> list[Pot]:
        committed = {s.seat: s.committed for s in self._seats.values() if s.in_hand and s.committed > 0}
        pots: list[Pot] = []
        prev = 0
        for level in sorted(set(committed.values())):
            amount = sum(min(c, level) - min(c, prev) for c in committed.values())
            eligible = tuple(s for s, c in committed.items() if c >= level and self._seats[s].contending)
            prev = level
            if not eligible and pots:  # dead money above every contender
                pots[-1] = Pot(pots[-1].amount + amount, pots[-1].eligible)
            elif pots and pots[-1].eligible == eligible:
                pots[-1] = Pot(pots[-1].amount + amount, eligible)
            else:
                pots.append(Pot(amount, eligible))
        return pots

    def _finish_hand(self) -> None:
        h = self._hand
        assert h is not None
        hero = self._cfg.hero_seat
        contenders = [s for s in self._seats.values() if s.contending]
        showdown = len(contenders) > 1
        ranks: dict[int, HandRank] = {}
        revealed: dict[int, tuple[Card, ...]] = {}
        if showdown:
            h.street, h.board_shown = Street.SHOWDOWN, 5
            ranks = {s.seat: evaluate_hand(s.hole + h.board_cards) for s in contenders}
            revealed = {s.seat: s.hole for s in contenders}
            self._emit(
                EventType.SHOWDOWN,
                {"board": list(h.board_cards),
                 "hands": {s: {"cards": list(c), "rank": ranks[s].name} for s, c in revealed.items()}},
                hand_id=h.hand_id,
            )

        winnings: dict[int, int] = {}
        awarded: list[Pot] = []
        for pot in self._build_pots():
            eligible = pot.eligible or tuple(s.seat for s in contenders)
            if showdown:
                best = max(ranks[s] for s in eligible)
                winners = [s for s in eligible if ranks[s] == best]
            else:
                winners = list(eligible)
            winners = self._clockwise(h.button, winners)  # odd chips go left of the button first
            share, odd = divmod(pot.amount, len(winners))
            for i, s in enumerate(winners):
                won = share + (1 if i < odd else 0)
                winnings[s] = winnings.get(s, 0) + won
                self._seats[s].stack += won
            awarded.append(Pot(pot.amount, pot.eligible, tuple(winners)))
            self._emit(
                EventType.POT_AWARDED,
                {"amount": pot.amount, "eligible": list(pot.eligible), "winners": winners,
                 "hand_rank": ranks[winners[0]].name if showdown else None},
                hand_id=h.hand_id,
            )

        if hero is not None and hero in h.starting_stacks:
            revealed.setdefault(hero, self._seats[hero].hole)
        final = {s: self._seats[s].stack for s in h.starting_stacks}
        result = HandResult(
            hand_id=h.hand_id,
            hand_number=h.number,
            table_id=self._cfg.table_id,
            button_seat=h.button,
            small_blind_seat=h.sb_seat,
            big_blind_seat=h.bb_seat,
            board=h.board,
            starting_stacks=dict(h.starting_stacks),
            final_stacks=final,
            blinds=dict(h.blinds),
            actions=tuple(h.actions),
            pots=tuple(awarded),
            winnings=winnings,
            net={s: final[s] - h.starting_stacks[s] for s in final},
            revealed_cards=revealed,
            hand_ranks={s: r.name for s, r in ranks.items()},
            went_to_showdown=showdown,
            started_at=h.started_at,
            ended_at=self._clock(),
        )
        self._history[h.hand_id] = result
        self._emit(
            EventType.HAND_ENDED,
            {"winnings": winnings, "net": result.net, "stacks": final, "showdown": showdown},
            hand_id=h.hand_id,
        )
        self._hand = None
        for s in h.starting_stacks:
            seat = self._seats[s]
            seat.bet = seat.committed = 0
            seat.in_hand = seat.folded = seat.all_in = False
            if seat.stack == 0:
                self._emit(EventType.PLAYER_BUSTED, {"seat": s, "player_id": seat.player_id}, hand_id=h.hand_id)
        if self._cfg.auto_deal:
            self._maybe_auto_deal()
