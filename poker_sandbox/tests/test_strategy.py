"""Tests for the M3 heuristic strategy (sync self-play on the M0 mock)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from adapter.config import AdapterConfig
from adapter.mock import MockAdapter
from adapter.models import Action, ActionType, Street, TableState
from agent.strategy import Strategy
from pokermath import parse_many
from strategy import (
    AGGRESSIVE,
    LOOSE,
    TIGHT,
    HeuristicParams,
    HeuristicStrategy,
    Position,
    Spot,
    board_texture,
    chart_for,
    collect,
    has_draw,
    play_hands,
    seat_positions,
)

A = ActionType
C = parse_many
PROFILES = {"tight": TIGHT, "loose": LOOSE, "aggressive": AGGRESSIVE}
STACK = 200  # 100 bb


def table(seats: int, seed: int = 1) -> MockAdapter:
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=seats, seed=seed, auto_deal=False, hero_seat=None))
    return mock


def mixed_strategies(seats: int, base_seed: int) -> tuple[dict[int, HeuristicStrategy], dict[int, str]]:
    groups = {s: list(PROFILES)[s % 3] for s in range(seats)}
    return {s: HeuristicStrategy(replace(PROFILES[groups[s]], seed=base_seed + s)) for s in range(seats)}, groups


# ------------------------------------------------------------- legality


def test_never_illegal_over_2000_hands() -> None:
    checked = 0

    def check(seat: int, state: TableState, action: Action) -> None:
        nonlocal checked
        checked += 1
        assert action.seat == seat and action.type in state.legal_actions, (state, action)
        if action.type in (A.BET, A.RAISE):
            assert state.min_raise_to <= action.amount <= state.max_raise_to
            assert action.amount - state.player(seat).bet <= state.player(seat).stack
        if A.CHECK in state.legal_actions:
            assert action.type is not A.FOLD

    hands = 0
    for seats, n in ((6, 1400), (2, 600)):
        strategies, _ = mixed_strategies(seats, 10)
        hands += len(play_hands(table(seats, seed=3), strategies, n, STACK, check))  # IllegalActionError would raise
    assert hands == 2000 and checked > 5000


def test_protocol_and_params_validation() -> None:
    assert isinstance(HeuristicStrategy(), Strategy)
    for bad in ({"aggression": 1.5}, {"bluff_freq": -0.1}, {"sizing_set": ()}, {"sizing_set": (0.5, -1.0)},
                {"vpip_scale": 0.0}, {"equity_samples": 10}):
        with pytest.raises(ValueError):
            HeuristicParams(**bad)


# ----------------------------------------------------------- determinism


def _checksum(seed: int) -> str:
    digest = hashlib.sha256()
    strategies, _ = mixed_strategies(6, seed)
    for r in play_hands(table(6, seed=5), strategies, 200, STACK):
        for rec in r.actions:
            digest.update(f"{r.hand_id}:{rec.seat}:{rec.type.value}:{rec.amount}|".encode())
    return digest.hexdigest()


def test_same_seed_same_actions() -> None:
    first = _checksum(7)
    assert first == _checksum(7)
    assert first != _checksum(8)


# --------------------------------------------------------------- profiles


def test_profiles_differ() -> None:
    strategies, groups = mixed_strategies(6, 20)
    results = play_hands(table(6, seed=9), strategies, 1500, STACK)
    stats = {g: s.ratios() for g, s in collect(results, groups, 2).items()}
    tight, loose, aggressive = stats["tight"], stats["loose"], stats["aggressive"]
    assert tight["vpip"] < loose["vpip"]
    assert aggressive["pfr"] > tight["pfr"]
    assert aggressive["3bet"] > tight["3bet"]
    assert aggressive["af"] > tight["af"]
    assert 8 < tight["vpip"] < 25 and 20 < loose["vpip"] < 50  # plausible human-like ranges


# ---------------------------------------------------------------- preflop


def _preflop_decision(hero_seat: int, hole: str, params: HeuristicParams, folds_before: int) -> Action:
    mock = table(6)
    mock.start_hand(button=0, hole_cards={hero_seat: C(hole)})  # SB=1, BB=2, UTG=3, MP=4, CO=5
    for _ in range(folds_before):
        seat = mock.view(None).to_act_seat
        assert seat is not None
        mock.send_action(Action(seat, A.FOLD))
    state = mock.view(hero_seat)
    assert state.to_act_seat == hero_seat
    return HeuristicStrategy(params).decide(state)


@pytest.mark.parametrize("hole", ["As Ah", "Ks Qs"])
def test_co_opens_strong_hands(hole: str) -> None:
    action = _preflop_decision(5, hole, HeuristicParams(seed=1), folds_before=2)
    assert action.type is A.RAISE and action.amount == 5  # 2.5 bb


def test_utg_folds_72o() -> None:
    for seed in range(20):
        assert _preflop_decision(3, "7h 2c", HeuristicParams(seed=seed), folds_before=0).type is A.FOLD


def test_charts_and_positions() -> None:
    assert seat_positions(table(6).view(None)) == {}  # no hand yet
    mock = table(6)
    mock.start_hand(button=0)
    assert seat_positions(mock.view(None)) == {
        1: Position.SB, 2: Position.BB, 3: Position.UTG, 4: Position.MP, 5: Position.CO, 0: Position.BTN}
    hu = table(2)
    hu.start_hand(button=1)
    assert seat_positions(hu.view(None)) == {1: Position.BTN, 0: Position.BB}
    utg, btn = chart_for(Position.UTG, Spot.OPEN), chart_for(Position.BTN, Spot.OPEN)
    assert utg.percent() < btn.percent()
    combos = set(btn.combos())
    assert all(pair in combos for pair in utg.combos())
    sb_vs_btn = chart_for(Position.SB, Spot.VS_OPEN, opener=Position.BTN)
    assert sb_vs_btn.percent() > chart_for(Position.SB, Spot.VS_OPEN).percent()


# --------------------------------------------------------------- postflop


def test_cbet_frequency_on_dry_board() -> None:
    params = HeuristicParams(cbet_freq=0.6, seed=42)
    hero = HeuristicStrategy(params)
    mock = table(2)
    mock.on_event(hero.on_event, seat=0)
    bets = trials = 0
    for _ in range(300):
        mock.set_stack(0, STACK)
        mock.set_stack(1, STACK)
        mock.start_hand(button=0, hole_cards={0: C("Ah Kd"), 1: C("9c 8c")}, board=C("Kc 7s 2d 4h Jh"))
        pre = hero.decide(mock.view(0))
        assert pre.type is A.RAISE
        mock.send_action(pre)
        mock.send_action(Action(1, A.CALL))
        assert mock.view(None).street is Street.FLOP
        mock.send_action(Action(1, A.CHECK))
        flop = hero.decide(mock.view(0))
        trials += 1
        bets += flop.type is A.BET
        mock.send_action(flop)
        while (seat := mock.view(None).to_act_seat) is not None:  # finish the hand passively
            st = mock.view(seat)
            mock.send_action(Action(seat, A.CHECK if A.CHECK in st.legal_actions else A.FOLD))
    assert board_texture(C("Kc 7s 2d")).wet is False
    assert bets / trials == pytest.approx(0.6, abs=0.1)


def test_folds_weak_hand_to_overbet() -> None:
    mock = table(2)
    hero = HeuristicStrategy(HeuristicParams(seed=3))
    mock.on_event(hero.on_event, seat=1)
    mock.start_hand(button=0, hole_cards={0: C("Ac Kc"), 1: C("3d 2h")}, board=C("As Kd Qh 8c 9c"))
    mock.send_action(Action(0, A.CALL))  # limp
    mock.send_action(Action(1, A.CHECK))
    mock.send_action(Action(1, A.CHECK))  # flop: BB acts first
    mock.send_action(Action(0, A.BET, 12))  # 3x pot
    state = mock.view(1)
    assert state.to_call > state.pot - state.to_call  # bet bigger than the pot before it
    assert not has_draw(C("3d 2h"), state.board)
    assert hero.decide(state).type is A.FOLD


def test_texture_and_draws() -> None:
    assert board_texture(C("9h Th 2c")).wet and board_texture(C("5c 6d 7h")).wet
    assert board_texture(C("Kc Kd 2h")).paired
    assert has_draw(C("Ah 5h"), C("Kh 9h 2c"))  # nut flush draw
    assert has_draw(C("8c 9d"), C("Th Jc 2s"))  # open-ended
    assert not has_draw(C("8c 9d"), C("Th Jc 2s 3d 4h"))  # river: no draws
