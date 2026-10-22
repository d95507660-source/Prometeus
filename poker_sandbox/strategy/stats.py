"""Player statistics from M0 hand histories: VPIP, PFR, 3-bet, AF, WTSD, winrate."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from adapter.models import ActionType, HandResult, Street

A = ActionType


@dataclass
class PlayerStats:
    """Counters for one player (or one profile spread over several seats)."""

    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    three_bet: int = 0
    three_bet_opps: int = 0
    postflop_aggr: int = 0
    postflop_calls: int = 0
    saw_flop: int = 0
    showdowns: int = 0
    net_bb: list[float] = field(default_factory=list)

    def ratios(self) -> dict[str, float]:
        """Percentages, AF and bb/100 with its standard error."""
        n = max(self.hands, 1)
        mean = sum(self.net_bb) / len(self.net_bb) if self.net_bb else 0.0
        var = (sum((x - mean) ** 2 for x in self.net_bb) / (len(self.net_bb) - 1)) if len(self.net_bb) > 1 else 0.0
        return {
            "hands": float(self.hands),
            "vpip": 100 * self.vpip / n,
            "pfr": 100 * self.pfr / n,
            "3bet": 100 * self.three_bet / max(self.three_bet_opps, 1),
            "af": self.postflop_aggr / self.postflop_calls if self.postflop_calls else float("inf"),
            "wtsd": 100 * self.showdowns / max(self.saw_flop, 1),
            "bb100": 100 * mean,
            "bb100_se": 100 * math.sqrt(var / len(self.net_bb)) if self.net_bb else 0.0,
        }


def collect(results: Iterable[HandResult], groups: Mapping[int, str], big_blind: int) -> dict[str, PlayerStats]:
    """Aggregate per group name (seat → group) over finished hands."""
    out = {g: PlayerStats() for g in groups.values()}
    for r in results:
        raises = 0
        voluntary: set[int] = set()
        raised: set[int] = set()
        three_bet: set[int] = set()
        three_bet_opp: set[int] = set()
        folded_pre: set[int] = set()
        for rec in r.actions:
            if rec.street is Street.PREFLOP:
                if raises == 1:
                    three_bet_opp.add(rec.seat)
                if rec.type in (A.CALL, A.BET, A.RAISE):
                    voluntary.add(rec.seat)
                if rec.type in (A.BET, A.RAISE):
                    raised.add(rec.seat)
                    raises += 1
                    if raises == 2:
                        three_bet.add(rec.seat)
                if rec.type is A.FOLD:
                    folded_pre.add(rec.seat)
            else:
                stats = out.get(groups.get(rec.seat, ""))
                if stats is None:
                    continue
                if rec.type in (A.BET, A.RAISE):
                    stats.postflop_aggr += 1
                elif rec.type is A.CALL:
                    stats.postflop_calls += 1
        flop_seen = len(r.board) >= 3
        for seat in r.starting_stacks:
            group = groups.get(seat)
            if group is None:
                continue
            s = out[group]
            s.hands += 1
            s.vpip += seat in voluntary
            s.pfr += seat in raised
            s.three_bet += seat in three_bet
            s.three_bet_opps += seat in three_bet_opp
            if flop_seen and seat not in folded_pre:
                s.saw_flop += 1
                s.showdowns += r.went_to_showdown and seat in r.hand_ranks
            s.net_bb.append(r.net[seat] / big_blind)
    return out
