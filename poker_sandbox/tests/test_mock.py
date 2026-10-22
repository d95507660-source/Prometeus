"""Unit tests for the offline NLHE engine behind MockAdapter."""

from __future__ import annotations

import random
import socket
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from adapter import (
    Action,
    ActionType,
    AdapterConfig,
    Event,
    EventType,
    HandNotFoundError,
    IllegalActionError,
    NotConnectedError,
    RateLimits,
    Stakes,
    Street,
    TableClosedError,
    cards,
)
from adapter.audit import JsonlAuditLog
from adapter.mock import MockAdapter, evaluate_hand

A = ActionType


def fixed_clock() -> Callable[[], datetime]:
    """Deterministic clock: +1 ms per call."""
    ticks: Iterator[int] = iter(range(10**9))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return lambda: start + timedelta(milliseconds=next(ticks))


def make(seats: int = 2, **overrides) -> MockAdapter:
    cfg = {"seats": seats, "seed": 123, "auto_deal": False, **overrides}
    adapter = MockAdapter(clock=fixed_clock())
    adapter.connect(AdapterConfig(**cfg))
    return adapter


def act(adapter: MockAdapter, kind: ActionType, amount: int = 0) -> None:
    state = adapter.view(None)
    assert state.to_act_seat is not None, "no one to act"
    adapter.send_action(Action(state.to_act_seat, kind, amount))


def check_down(adapter: MockAdapter) -> None:
    while (state := adapter.view(None)).to_act_seat is not None:
        act(adapter, A.CHECK if A.CHECK in state.legal_actions else A.CALL)


def random_bot_session(seed: int, seats: int, log_path: Path | None = None, hands: int = 40) -> MockAdapter:
    adapter = MockAdapter(clock=fixed_clock())
    adapter.connect(
        AdapterConfig(seats=seats, seed=seed, log_path=log_path, limits=RateLimits(max_hands=hands))
    )
    rng = random.Random(seed)
    total = seats * 200  # default Stakes.starting_stack
    while (state := adapter.view(None)).to_act_seat is not None:
        kind = rng.choice(state.legal_actions)
        amount = rng.randint(state.min_raise_to, state.max_raise_to) if kind in (A.BET, A.RAISE) else 0
        adapter.send_action(Action(state.to_act_seat, kind, amount, hand_id=state.hand_id))
        after = adapter.view(None)
        assert sum(p.stack + p.committed for p in after.players) == total
        assert all(p.stack >= 0 for p in after.players)
    return adapter


# --------------------------------------------------------------------- evaluator


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        ("9h Th Jh Qh Kh", "As Ad Ac Ah Kd"),  # straight flush > quads
        ("As Ad Ac Ah 2d", "Ks Kd Kc Qh Qd"),  # quads > full house
        ("Ks Kd Kc 2h 2d", "Ah Qh 9h 5h 3h"),  # full house > flush
        ("Ah Qh 9h 5h 3h", "Td Jc Qs Kh As"),  # flush > straight
        ("2d 3c 4s 5h 6s", "Ad 2c 3s 4h 5s"),  # 6-high straight > wheel
        ("Ad 2c 3s 4h 5s", "Qs Qd Qc 2h 3d"),  # wheel > trips
        ("Qs Qd Qc 2h 3d", "As Ad Ks Kd Qh"),  # trips > two pair
        ("As Ad Ks Kd 2h", "As Ad Qs Qd Kh"),  # higher second pair
        ("As Ad Ks Kd 3h", "As Ad Ks Kd 2h"),  # two-pair kicker
        ("9s 9d Ah 7c 2d", "9s 9d Kh Qc Jd"),  # pair kicker
        ("Ah Kd 9c 7s 3d", "Ah Kd 9c 7s 2d"),  # high card, last kicker
    ],
)
def test_hand_ordering(stronger: str, weaker: str) -> None:
    assert evaluate_hand(cards(stronger)) > evaluate_hand(cards(weaker))


def test_evaluator_picks_best_five_of_seven() -> None:
    rank = evaluate_hand(cards("Ah Kh 2h 7h 9h 9d 9c"))
    assert rank.name == "Flush"
    assert rank.tiebreak == (14, 13, 9, 7, 2)
    full = evaluate_hand(cards("Ks Kd Kc 7h 7d 7s 2c"))
    assert full.name == "Full House" and full.tiebreak == (13, 7)


def test_evaluator_equal_hands_tie() -> None:
    board = cards("5h 6d 7c 8s 9h")
    assert evaluate_hand(cards("2c 3d") + board) == evaluate_hand(cards("2h 3s") + board)


def test_evaluator_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        evaluate_hand(cards("Ah Kh 2h 7h"))
    with pytest.raises(ValueError):
        evaluate_hand(cards("Ah Ah 2h 7h 9h"))


# ------------------------------------------------------------- blinds & order


def test_heads_up_button_posts_small_blind_and_acts_first_preflop() -> None:
    a = make(2)
    a.start_hand(button=0)
    st = a.view(None)
    assert (st.player(0).bet, st.player(1).bet) == (1, 2)
    assert st.to_act_seat == 0 and st.street is Street.PREFLOP
    act(a, A.CALL)
    st = a.view(None)
    assert st.to_act_seat == 1
    assert A.RAISE in st.legal_actions  # big blind option
    act(a, A.CHECK)
    st = a.view(None)
    assert st.street is Street.FLOP and len(st.board) == 3
    assert st.to_act_seat == 1  # BB (non-button) acts first postflop


def test_big_blind_option_reopens_action() -> None:
    a = make(2)
    a.start_hand(button=0)
    act(a, A.CALL)
    act(a, A.RAISE, 6)
    st = a.view(None)
    assert st.to_act_seat == 0 and st.street is Street.PREFLOP and st.to_call == 4


def test_six_max_positions_and_button_rotation() -> None:
    a = make(6)
    a.start_hand(button=4)
    hid = a.view(None).hand_id
    st = a.view(None)
    assert st.player(5).bet == 1 and st.player(0).bet == 2
    assert st.to_act_seat == 1  # UTG
    while a.view(None).to_act_seat is not None:  # fold around to the big blind
        act(a, A.FOLD)
    result = a.get_hand_history(hid)
    assert result.net == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: -1}
    a.start_hand()
    st = a.view(None)
    assert st.button_seat == 5 and st.player(0).bet == 1 and st.player(1).bet == 2
    assert st.to_act_seat == 2


def test_button_skips_busted_seat() -> None:
    a = make(3)
    a.set_stack(1, 0)
    a.start_hand(button=0)
    st = a.view(None)
    assert not st.player(1).in_hand
    # heads-up between 0 and 2: button 0 is small blind
    assert st.player(0).bet == 1 and st.player(2).bet == 2


# ----------------------------------------------------------- pots & showdown


def test_everyone_folds_uncalled_raise_returned() -> None:
    a = make(3)
    hid = a.start_hand(button=0)  # sb=1, bb=2, UTG=0
    act(a, A.RAISE, 6)
    act(a, A.FOLD)
    act(a, A.FOLD)
    r = a.get_hand_history(hid)
    assert r.net == {0: 3, 1: -1, 2: -2}
    assert not r.went_to_showdown
    assert r.board == ()
    assert sum(p.amount for p in r.pots) == 9


def test_showdown_best_hand_wins() -> None:
    a = make(2)
    hid = a.start_hand(
        button=0, hole_cards={0: cards("Kc Kd"), 1: cards("Ac Ad")}, board=cards("2c 7d 9h Js 3s")
    )
    check_down(a)
    r = a.get_hand_history(hid)
    assert r.went_to_showdown
    assert r.net == {0: -2, 1: 2}
    assert r.hand_ranks == {0: "One Pair", 1: "One Pair"}
    assert r.revealed_cards[0] == cards("Kc Kd") and r.revealed_cards[1] == cards("Ac Ad")
    assert r.board == cards("2c 7d 9h Js 3s")
    assert [x.street for x in r.actions] == [
        Street.PREFLOP, Street.PREFLOP, Street.FLOP, Street.FLOP,
        Street.TURN, Street.TURN, Street.RIVER, Street.RIVER,
    ]


def test_split_pot_odd_chip_goes_left_of_button() -> None:
    a = make(3)
    holes = {0: cards("2c 3d"), 1: cards("2h 3s"), 2: cards("2d 3c")}
    hid = a.start_hand(button=0, hole_cards=holes, board=cards("5h 6d 7c 8s 9h"))
    act(a, A.CALL)   # seat 0 (button) calls 2
    act(a, A.FOLD)   # seat 1 (SB) folds, 1 chip dead
    act(a, A.CHECK)  # seat 2 (BB)
    check_down(a)
    r = a.get_hand_history(hid)
    assert r.pots[0].amount == 5 and set(r.pots[0].winners) == {0, 2}
    assert r.winnings == {2: 3, 0: 2}  # seat 2 is first clockwise from the button


def test_side_pots_with_three_all_ins() -> None:
    a = make(3)
    for seat, stack in {0: 50, 1: 100, 2: 200}.items():
        a.set_stack(seat, stack)
    holes = {0: cards("As Ah"), 1: cards("Ks Kh"), 2: cards("Qs Qh")}
    hid = a.start_hand(button=0, hole_cards=holes, board=cards("2c 5d 8h Jc 3d"))
    act(a, A.ALL_IN)  # seat 0: 50
    act(a, A.ALL_IN)  # seat 1: 100
    act(a, A.CALL)    # seat 2 calls 100
    assert a.view(None).hand_id is None  # board ran out automatically
    r = a.get_hand_history(hid)
    assert [(p.amount, p.eligible, p.winners) for p in r.pots] == [
        (150, (0, 1, 2), (0,)),
        (100, (1, 2), (1,)),
    ]
    assert r.final_stacks == {0: 150, 1: 100, 2: 100}
    assert len(r.board) == 5


def test_uncalled_all_in_excess_returned() -> None:
    a = make(2)
    a.set_stack(0, 50)
    hid = a.start_hand(button=0, hole_cards={0: cards("Ac Ad"), 1: cards("7c 2d")},
                       board=cards("Kh Qs 9d 4c 3h"))
    act(a, A.CALL)
    act(a, A.ALL_IN)  # BB shoves 200
    act(a, A.CALL)    # button calls all-in for 50 total
    r = a.get_hand_history(hid)
    assert r.final_stacks == {0: 100, 1: 150}
    assert r.pots[-1].eligible == (1,) and r.pots[-1].amount == 150


# ------------------------------------------------------------ betting rules


def test_illegal_actions_are_rejected_and_published() -> None:
    a = make(2)
    rejected: list[Event] = []
    a.on_event(lambda e: rejected.append(e) if e.type is EventType.ACTION_REJECTED else None)
    hid = a.start_hand(button=0)
    bad = [
        Action(1, A.CALL),                     # out of turn
        Action(0, A.CHECK),                    # facing the big blind
        Action(0, A.RAISE, 3),                 # below min-raise (to 4)
        Action(0, A.RAISE, 500),               # more than the stack
        Action(0, A.BET, 10),                  # there is already a bet
        Action(0, A.CALL, hand_id="other-1"),  # stale hand id
    ]
    for action in bad:
        with pytest.raises(IllegalActionError):
            a.send_action(action)
    assert len(rejected) == len(bad)
    assert all(e.hand_id == hid and e.payload["reason"] for e in rejected)
    act(a, A.CALL)
    with pytest.raises(IllegalActionError, match="fold"):
        a.send_action(Action(1, A.FOLD))  # checking is free
    a.send_action(Action(1, A.RAISE, 4))  # exact min-raise is fine


def test_incomplete_all_in_raise_does_not_reopen_betting() -> None:
    a = make(3)
    a.set_stack(1, 14)
    a.start_hand(button=0)  # sb=1 (14 chips), bb=2, UTG=0
    act(a, A.RAISE, 10)      # full raise of 8
    act(a, A.ALL_IN)         # SB to 14: raise of 4 < 8, incomplete
    st = a.view(None)
    assert st.to_act_seat == 2 and A.RAISE in st.legal_actions
    assert st.min_raise_to == 22  # 14 + last full raise (8)
    act(a, A.CALL)
    st = a.view(None)
    assert st.to_act_seat == 0
    assert st.legal_actions == (A.FOLD, A.CALL)
    with pytest.raises(IllegalActionError, match="reopened"):
        a.send_action(Action(0, A.RAISE, 40))
    act(a, A.CALL)
    assert a.view(None).street is Street.FLOP


def test_full_raise_reopens_betting() -> None:
    a = make(3)
    a.start_hand(button=0)
    act(a, A.RAISE, 10)
    act(a, A.CALL)
    act(a, A.RAISE, 30)  # BB re-raises by 20
    st = a.view(None)
    assert st.to_act_seat == 0 and A.RAISE in st.legal_actions and st.min_raise_to == 50


def test_short_big_blind_all_in() -> None:
    a = make(2)
    a.set_stack(1, 1)
    hid = a.start_hand(button=0)  # BB has 1 chip, posts it all-in
    st = a.view(None)
    assert st.player(1).is_all_in
    act(a, A.CALL)  # button completes to the full big blind
    r = a.get_hand_history(hid)
    assert sum(r.final_stacks.values()) == 201
    assert sum(p.amount for p in r.pots) == 3


# ------------------------------------------------------- info, events, logs


def test_information_is_filtered_per_seat() -> None:
    a = make(6, hero_seat=2)
    hero_events: list[Event] = []
    seat4_events: list[Event] = []
    a.on_event(hero_events.append)
    a.on_event(seat4_events.append, seat=4)
    hid = a.start_hand(button=0)
    st = a.get_table_state()
    assert st.player(2).hole_cards is not None
    assert all(p.hole_cards is None for p in st.players if p.seat != 2)
    hero_holes = [e for e in hero_events if e.type is EventType.HOLE_CARDS]
    assert [e.recipient_seat for e in hero_holes] == [2]
    assert [e.recipient_seat for e in seat4_events if e.type is EventType.HOLE_CARDS] == [4]
    assert not any(e.type in (EventType.DECK, EventType.SESSION_STARTED) for e in hero_events)
    while a.view(None).to_act_seat is not None:
        act(a, A.FOLD)
    assert set(a.get_hand_history(hid).revealed_cards) == {2}


def test_jsonl_audit_log(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    a = random_bot_session(7, 3, log_path=log, hands=5)
    a.disconnect()
    cfg = AdapterConfig(seats=2, api_token="super-secret", log_path=tmp_path / "t.jsonl")
    b = MockAdapter()
    with b.session(cfg):
        pass
    assert "super-secret" not in (tmp_path / "t.jsonl").read_text()

    records = list(JsonlAuditLog.read(log))
    assert records[0]["type"] == "session_started" and records[0]["payload"]["seed"] == 7
    assert [r["seq"] for r in records] == list(range(1, len(records) + 1))
    assert all("timestamp" in r for r in records)
    decks = [r for r in records if r["type"] == "deck"]
    assert len(decks) == 5 and all(r["visibility"] == "audit" for r in decks)
    logged_actions = sum(1 for r in records if r["type"] == "action")
    assert logged_actions == sum(len(a.get_hand_history(h).actions) for h in a.hand_ids)
    assert records[-1]["type"] == "disconnected"


def test_same_seed_reproduces_session(tmp_path: Path) -> None:
    random_bot_session(42, 6, log_path=tmp_path / "a.jsonl").disconnect()
    random_bot_session(42, 6, log_path=tmp_path / "b.jsonl").disconnect()
    random_bot_session(43, 6, log_path=tmp_path / "c.jsonl").disconnect()
    # Line 1 (session header) differs only by log_path; everything after must match.
    a, b, c = ((tmp_path / f"{n}.jsonl").read_text().splitlines()[1:] for n in "abc")
    assert a == b
    assert a != c


@pytest.mark.parametrize("seats", [2, 3, 6])
@pytest.mark.parametrize("seed", range(15))
def test_chip_conservation_with_random_bots(seats: int, seed: int) -> None:
    a = random_bot_session(seed, seats)
    total = seats * 200
    assert sum(p.stack for p in a.view(None).players) == total
    for hid in a.hand_ids:
        r = a.get_hand_history(hid)
        assert sum(r.net.values()) == 0
        assert sum(r.winnings.values()) == sum(p.amount for p in r.pots)
        assert len(set(r.board)) == len(r.board)


def test_callback_driven_bots_play_until_hand_limit() -> None:
    a = MockAdapter(clock=fixed_clock())
    ended: list[Event] = []
    idle: list[Event] = []

    def make_bot(seat: int):
        def on_event(e: Event) -> None:
            if e.type is EventType.TURN and e.payload["seat"] == seat:
                kind = A.CHECK if A.CHECK in e.payload["legal_actions"] else A.CALL
                a.send_action(Action(seat, kind, hand_id=e.hand_id))
        return on_event

    for seat in range(4):
        a.on_event(make_bot(seat), seat=seat)
    a.on_event(lambda e: (ended if e.type is EventType.HAND_ENDED else idle).append(e)
               if e.type in (EventType.HAND_ENDED, EventType.TABLE_IDLE) else None)
    a.connect(AdapterConfig(seats=4, seed=1, limits=RateLimits(max_hands=5)))
    assert len(ended) == 5 and len(idle) == 1
    assert all(ev.payload["showdown"] for ev in ended)
    with pytest.raises(TableClosedError):
        a.start_hand()


def test_mock_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    random_bot_session(3, 6, hands=10).disconnect()


def test_lifecycle_errors() -> None:
    a = MockAdapter()
    with pytest.raises(NotConnectedError):
        a.get_table_state()
    with pytest.raises(NotConnectedError):
        a.send_action(Action(0, A.CHECK))
    a.connect(AdapterConfig(seats=2, seed=1, auto_deal=False))
    with pytest.raises(HandNotFoundError):
        a.get_hand_history("nope")
    with pytest.raises(IllegalActionError, match="no hand"):
        a.send_action(Action(0, A.CHECK))
    a.start_hand()
    with pytest.raises(TableClosedError):
        a.start_hand()
    a.disconnect()
    a.disconnect()  # idempotent
    assert not a.is_connected


def test_unsubscribe_stops_delivery() -> None:
    a = make(2)
    got: list[Event] = []
    unsubscribe = a.on_event(got.append)
    a.start_hand()
    n = len(got)
    unsubscribe()
    act(a, A.FOLD)
    assert len(got) == n


def test_random_seed_is_recorded_when_not_given() -> None:
    a = MockAdapter()
    a.connect(AdapterConfig(seats=2, auto_deal=False))
    assert isinstance(a.seed, int)


# ------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seats": 7},
        {"seats": 1},
        {"seats": 2, "hero_seat": 2},
        {"stakes": {"small_blind": 5, "big_blind": 2}},
        {"seats": 2, "player_ids": ["a"]},
        {"seats": 2, "player_ids": ["a", "a"]},
        {"table_id": "bad id/../"},
        {"unknown": 1},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        AdapterConfig(**kwargs)


def test_config_redacts_token() -> None:
    cfg = AdapterConfig(base_url="https://sandbox.local/api", api_token="tok", stakes=Stakes(big_blind=4))
    dumped = cfg.redacted()
    assert dumped["api_token"] == "**********"
    assert dumped["stakes"]["big_blind"] == 4
