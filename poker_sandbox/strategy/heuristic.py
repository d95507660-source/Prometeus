"""Chart + equity heuristic strategy for a single bot. Aims at plausible play, not maximal EV.

Preflop: position from the button, chart from ``charts.py`` scaled by
vpip/pfr, boundary classes mixed by weight. Postflop: Monte Carlo equity vs an
estimated villain range, pot odds, board texture and draws drive the choice.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import pairwise

from adapter.models import Action, ActionType, Card, Event, EventType, HandResult, PlayerState, Street, TableState
from agent.strategy import BaseStrategy, passive_or_fold
from pokermath.equity import equity_vs_range
from pokermath.odds import required_equity
from pokermath.ranges import Range
from strategy.charts import (
    OPEN_CHARTS,
    VS_4BET_CHART,
    Position,
    Spot,
    SpotChart,
    classes_of,
    hand_class,
    parse_cached,
    scale_classes,
    spot_chart,
)
from strategy.params import HeuristicParams

A = ActionType
_TAIL_POSITIONS = (Position.UTG, Position.MP, Position.CO, Position.BTN)


def seat_positions(state: TableState) -> dict[int, Position]:
    """Map seats dealt into the hand to 6-max positions (shorter tables drop UTG first)."""
    if state.button_seat is None:
        return {}
    seats = [p.seat for p in state.players if p.in_hand]
    if len(seats) < 2:
        return {}
    btn, n = state.button_seat, state.max_seats
    order = sorted(seats, key=lambda s: (s - btn - 1) % n)  # SB first, button last
    if len(order) == 2:
        return {btn: Position.BTN, next(s for s in order if s != btn): Position.BB}
    tail = _TAIL_POSITIONS[-(len(order) - 2):]  # M0 tables have at most 6 seats
    return dict(zip(order, (Position.SB, Position.BB, *tail), strict=True))


@dataclass(frozen=True, slots=True)
class BoardTexture:
    """Coarse board description used for bet/bluff decisions."""

    max_suit: int
    paired: bool
    connected: bool

    @property
    def wet(self) -> bool:
        """Monotone/two-tone-and-close or straighty boards."""
        return self.max_suit >= 3 or self.connected


def _rank_set(cards: tuple[Card, ...]) -> set[int]:
    ranks = {c.rank for c in cards}
    return ranks | {1} if 14 in ranks else ranks  # ace also plays low


def board_texture(board: tuple[Card, ...]) -> BoardTexture:
    """Suits, pairing and connectedness of the board."""
    suits = [sum(c.suit == s for c in board) for s in "cdhs"]
    ranks = _rank_set(board)
    connected = any(sum(r in ranks for r in range(lo, lo + 5)) >= 3 for lo in range(1, 11))
    distinct = sorted({c.rank for c in board})
    close_two_tone = max(suits) == 2 and any(b - a <= 2 for a, b in pairwise(distinct))
    return BoardTexture(max(suits), len(distinct) < len(board), connected or close_two_tone)


def has_draw(hole: tuple[Card, ...], board: tuple[Card, ...]) -> bool:
    """Flush draw or straight draw (open-ended or gutshot) that uses a hole card; never on the river."""
    if len(board) >= 5:
        return False
    cards = hole + board
    for suit in {c.suit for c in hole}:
        if sum(c.suit == suit for c in cards) == 4:
            return True
    ranks, hole_ranks = _rank_set(cards), _rank_set(hole)
    for lo in range(1, 11):
        window = set(range(lo, lo + 5))
        if len(window & ranks) == 4 and window & hole_ranks:
            return True
    return False


@dataclass(slots=True)
class _HandTracker:
    """Preflop history rebuilt from ACTION events (more precise than inferring from bets)."""

    hand_id: str | None = None
    preflop_raises: int = 0
    preflop_aggressor: int | None = None


class HeuristicStrategy(BaseStrategy):
    """Parametrised chart/equity heuristic.

    Feed it table events through ``on_event`` (the M1 Agent does) to know the
    preflop aggressor; without events preflop raises are inferred from bets
    and c-bet logic is skipped.
    """

    def __init__(self, params: HeuristicParams | None = None, *, name: str | None = None) -> None:
        self.params = params or HeuristicParams()
        self._rng = random.Random(self.params.seed)
        self._tracker = _HandTracker()
        self._name = name or type(self).__name__

    @property
    def name(self) -> str:
        """Strategy name for logs."""
        return self._name

    # ---------------------------------------------------------------- hooks

    def on_event(self, event: Event) -> None:
        """Track preflop raises and the aggressor of the current hand."""
        if event.type is EventType.HAND_STARTED:
            self._tracker = _HandTracker(hand_id=event.hand_id)
        elif event.type is EventType.ACTION and event.hand_id == self._tracker.hand_id:
            street = Street(event.payload["street"])
            kind = ActionType(event.payload["type"])
            if street is Street.PREFLOP and kind in (A.BET, A.RAISE):
                self._tracker.preflop_raises += 1
                self._tracker.preflop_aggressor = int(event.payload["seat"])

    def on_hand_end(self, result: HandResult) -> None:
        """Nothing to learn: the heuristic is stateless across hands."""

    def decide(self, state: TableState) -> Action:
        """Pick a legal action for ``state.to_act_seat``."""
        seat = state.to_act_seat
        if seat is None:
            raise ValueError("nobody is to act")
        hole = state.player(seat).hole_cards
        positions = seat_positions(state)
        if not hole or seat not in positions:
            return passive_or_fold(state)
        if state.street is Street.PREFLOP:
            action = self._preflop(state, seat, hole, positions)
        else:
            action = self._postflop(state, seat, hole, positions)
        return self._legalize(state, action)

    # ------------------------------------------------------------- preflop

    def _mix(self, weight: float) -> bool:
        if weight >= 1.0:
            return True
        if weight <= 0.0:
            return False
        return self._rng.random() < weight

    def _preflop_history(self, state: TableState) -> tuple[int, int | None]:
        if self._tracker.hand_id == state.hand_id and state.hand_id is not None:
            return self._tracker.preflop_raises, self._tracker.preflop_aggressor
        # Fallback without events: every distinct bet level above the big blind is one raise.
        levels = {p.bet for p in state.players if p.bet > state.big_blind}
        if not levels:
            return 0, None
        top = max(state.players, key=lambda p: p.bet)
        return len(levels), top.seat

    def _preflop(self, state: TableState, seat: int, hole: tuple[Card, ...],
                 positions: dict[int, Position]) -> Action:
        p = self.params
        label = hand_class(hole[0], hole[1])
        pos = positions[seat]
        raises, aggressor = self._preflop_history(state)
        bb = state.big_blind
        if raises == 0:
            chart = OPEN_CHARTS[pos]
            limpers = sum(1 for q in state.players if q.in_hand and q.bet == bb and positions.get(q.seat)
                          not in (Position.BB,))
            size = (4 if pos is Position.BB else 3 if pos is Position.SB else 2.5) * bb + limpers * bb
            if self._mix(scale_classes(classes_of(chart.raise_value), p.pfr_scale).get(label, 0.0)):
                return self._raise_to(state, round(size))
            limp = scale_classes(classes_of(chart.raise_value), p.vpip_scale).get(label, 0.0)
            if p.vpip_scale > p.pfr_scale and self._mix(limp):
                return Action(seat, A.CALL if state.to_call else A.CHECK)
            return Action(seat, A.FOLD if state.to_call else A.CHECK)

        if raises == 1:
            opener = positions.get(aggressor) if aggressor is not None else None
            spot = Spot.BB_DEFEND if pos is Position.BB else Spot.VS_OPEN
            chart = spot_chart(pos, spot, opener)
            oop = pos in (Position.SB, Position.BB)
            size = state.current_bet * (3.5 if oop else 3.0)
        elif raises == 2:
            chart = spot_chart(pos, Spot.VS_3BET)
            size = state.current_bet * 2.3
        else:
            chart = VS_4BET_CHART
            size = state.max_raise_to  # 5-bet = jam
        return self._facing_raise(state, seat, label, chart, size)

    def _facing_raise(self, state: TableState, seat: int, label: str, chart: SpotChart, size: float) -> Action:
        p = self.params
        if self._mix(scale_classes(classes_of(chart.raise_value), p.pfr_scale).get(label, 0.0)):
            return self._raise_to(state, round(size))
        if label in classes_of(chart.raise_bluff) and self._mix(min(1.0, 4 * p.bluff_freq)):
            return self._raise_to(state, round(size))
        continuing = classes_of(chart.raise_value) | classes_of(chart.call)
        if self._mix(scale_classes(continuing, p.vpip_scale).get(label, 0.0)):
            return Action(seat, A.CALL)
        return Action(seat, A.FOLD)

    # ------------------------------------------------------------ postflop

    def _villain_range(self, state: TableState, villain: int, positions: dict[int, Position]) -> Range:
        pos = positions.get(villain, Position.BTN)
        t = self._tracker
        known = t.hand_id == state.hand_id
        if not known or t.preflop_raises == 0:
            chart = OPEN_CHARTS[pos]
            return parse_cached(chart.raise_value)
        spot = Spot.BB_DEFEND if pos is Position.BB else Spot.VS_OPEN
        if t.preflop_aggressor == villain:
            if t.preflop_raises == 1:
                return parse_cached(OPEN_CHARTS[pos].raise_value)
            chart = spot_chart(pos, spot)
            return parse_cached(", ".join(s for s in (chart.raise_value, chart.raise_bluff) if s))
        chart = spot_chart(pos, spot) if t.preflop_raises == 1 else spot_chart(pos, Spot.VS_3BET)
        return parse_cached(chart.call or chart.raise_value)

    def _equity(self, state: TableState, seat: int, hole: tuple[Card, ...],
                opponents: list[PlayerState], positions: dict[int, Position]) -> float:
        t = self._tracker
        aggressor_in = t.hand_id == state.hand_id and any(o.seat == t.preflop_aggressor for o in opponents)
        villain = t.preflop_aggressor if aggressor_in and t.preflop_aggressor is not None else \
            max(opponents, key=lambda o: o.committed).seat
        try:
            eq = equity_vs_range(hole, state.board, self._villain_range(state, villain, positions),
                                 self.params.equity_samples, self._rng.getrandbits(31))
        except ValueError:  # villain range fully blocked by the board
            eq = 0.5
        return eq ** len(opponents)  # crude multiway discount, independent opponents assumed

    def _postflop(self, state: TableState, seat: int, hole: tuple[Card, ...],
                  positions: dict[int, Position]) -> Action:
        p = self.params
        agg = p.aggression
        opponents = [o for o in state.players if o.in_hand and not o.has_folded and o.seat != seat]
        if not opponents:
            return Action(seat, A.CHECK)
        texture = board_texture(state.board)
        draw = has_draw(hole, state.board)
        t = self._tracker

        if state.to_call == 0:
            cbet_spot = (state.street is Street.FLOP and t.hand_id == state.hand_id
                         and t.preflop_raises == 1 and t.preflop_aggressor == seat)
            if cbet_spot:  # frequency-driven by design, regardless of hand strength
                p_cbet = p.cbet_freq * (0.85 if texture.wet else 1.0)
                return self._bet(state, dry=not texture.wet) if self._mix(p_cbet) else Action(seat, A.CHECK)
            eq = self._equity(state, seat, hole, opponents, positions)
            if eq >= 0.62 - 0.12 * agg and self._mix(0.55 + 0.4 * agg):
                return self._bet(state, dry=not texture.wet)
            if draw and self._mix(0.25 + 0.4 * agg):
                return self._bet(state, dry=False)
            if self._mix(p.bluff_freq * (1.3 if not texture.wet else 0.7)):
                return self._bet(state, dry=not texture.wet)
            return Action(seat, A.CHECK)

        eq = self._equity(state, seat, hole, opponents, positions)
        needed = required_equity(state.pot, state.to_call)
        effective = eq + (0.06 if draw else 0.0)  # rough implied-odds credit for draws
        can_raise = A.RAISE in state.legal_actions
        if can_raise and effective >= 0.82 - 0.12 * agg and self._mix(0.35 + 0.5 * agg):
            return self._raise_postflop(state)
        if effective >= needed:
            return Action(seat, A.CALL)
        return Action(seat, A.FOLD)

    # ------------------------------------------------------------- sizing

    def _pick_fraction(self, dry: bool) -> float:
        sizes = sorted(self.params.sizing_set)
        if len(sizes) == 1:
            return sizes[0]
        tilt = 2.5 * (2 * self.params.aggression - 1) - (0.5 if dry else 0.0)  # dry boards: smaller bets
        weights = [math.exp(tilt * i / (len(sizes) - 1)) for i in range(len(sizes))]
        return self._rng.choices(sizes, weights=weights)[0]

    def _bet(self, state: TableState, *, dry: bool) -> Action:
        frac = self._pick_fraction(dry)
        hero_bet = state.player(state.to_act_seat or 0).bet
        return self._raise_to(state, hero_bet + round(frac * state.pot))

    def _raise_postflop(self, state: TableState) -> Action:
        frac = self._pick_fraction(dry=False)
        return self._raise_to(state, state.current_bet + round(frac * (state.pot + state.to_call)))

    def _raise_to(self, state: TableState, amount: int) -> Action:
        seat = state.to_act_seat or 0
        kind = A.BET if A.BET in state.legal_actions else A.RAISE
        return Action(seat, kind, amount)

    def _legalize(self, state: TableState, action: Action) -> Action:
        """Map any intent onto ``legal_actions`` and clamp amounts to the stack."""
        legal = state.legal_actions
        seat = action.seat
        kind = action.type
        if kind in (A.BET, A.RAISE):
            aggressive = A.BET if A.BET in legal else A.RAISE if A.RAISE in legal else None
            if aggressive is None:
                kind = A.CALL if A.CALL in legal else A.CHECK
            else:
                amount = max(state.min_raise_to, min(action.amount, state.max_raise_to))
                if amount >= 0.85 * state.max_raise_to:
                    amount = state.max_raise_to  # don't leave a sliver behind
                return Action(seat, aggressive, amount, hand_id=state.hand_id)
        if kind is A.CALL and A.CALL not in legal:
            kind = A.CHECK
        if kind is A.FOLD and A.CHECK in legal:
            kind = A.CHECK  # never fold when checking is free
        if kind not in legal:
            return passive_or_fold(state)
        return Action(seat, kind, hand_id=state.hand_id)
