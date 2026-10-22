"""Planned decision delays: lognormal noise * complexity * fatigue, plus occasional tanks.

Nothing here sleeps: delays are numbers written to the behaviour log for detectors.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from adapter.models import Street, TableState
from behavior.profiles import HumanLikeProfile
from pokermath.odds import required_equity

__all__ = ["DecisionContext", "TimingModel", "TimingReport", "context_from_state", "timing_report"]

_STREET_WEIGHT = {Street.PREFLOP: 0.0, Street.FLOP: 0.1, Street.TURN: 0.2, Street.RIVER: 0.4}
_BIG_DECISION = 0.5  # complexity at/above which a tank is possible
_CLOSE_EQUITY_MARGIN = 0.07
_CLOSE_PRICE = (0.2, 0.45)  # required equity of a "meaningful" call when the delegate gives no equity


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """What the timing model needs to know about one decision.

    Attributes:
        street: Current street (``None`` is treated as preflop).
        pot_bb: Pot before hero acts, in big blinds.
        to_call_bb: Amount to call, in big blinds.
        is_close_decision: Equity near the break-even threshold.
        session_pos: 0 at session start, 1 at its end (fatigue).
        tilted: Player is on tilt.
    """

    street: Street | None = Street.PREFLOP
    pot_bb: float = 1.5
    to_call_bb: float = 0.0
    is_close_decision: bool = False
    session_pos: float = 0.0
    tilted: bool = False

    @property
    def complexity(self) -> float:
        """0 (routine) .. 1 (river, big bet into a big pot, close spot)."""
        score = _STREET_WEIGHT.get(self.street or Street.PREFLOP, 0.0)
        if self.pot_bb > 0:
            score += min(0.4, 0.4 * self.to_call_bb / self.pot_bb)
        if self.pot_bb >= 20:
            score += 0.2
        if self.is_close_decision:
            score += 0.3
        return min(1.0, score)


def context_from_state(state: TableState, *, session_pos: float = 0.0, tilted: bool = False,
                       equity: float | None = None) -> DecisionContext:
    """Build a context from a table state; ``equity`` (if the delegate exposes it) sharpens ``is_close``."""
    bb = state.big_blind
    close = False
    if state.to_call > 0 and state.pot + state.to_call > 0:
        needed = required_equity(state.pot, state.to_call)
        if equity is not None:
            close = abs(equity - needed) < _CLOSE_EQUITY_MARGIN
        else:
            close = _CLOSE_PRICE[0] <= needed <= _CLOSE_PRICE[1]
    return DecisionContext(street=state.street, pot_bb=state.pot / bb, to_call_bb=state.to_call / bb,
                           is_close_decision=close, session_pos=min(1.0, max(0.0, session_pos)), tilted=tilted)


class TimingModel:
    """Samples planned delays. Stateless: all randomness comes from the passed ``rng``."""

    @staticmethod
    def sample(profile: HumanLikeProfile, ctx: DecisionContext, rng: random.Random) -> float:
        """Planned delay in ms; equals ``base_latency_ms`` exactly at ``humanlike_level == 0``."""
        # Always draw the same three numbers so streams stay aligned across levels and tilt states.
        z = rng.normalvariate(0.0, 1.0)
        u_tank, u_size = rng.random(), rng.random()
        level = profile.humanlike_level
        if level == 0.0:
            return profile.base_latency_ms
        complexity = ctx.complexity
        fatigue = 1.0 + level * profile.fatigue_drift * ctx.session_pos
        sigma = level * profile.latency_sigma * fatigue
        if ctx.tilted:
            sigma *= 1.0 + profile.tilt_extra_variance
        complexity_mult = 1.0 + level * (profile.complexity_factor - 1.0) * complexity
        delay = profile.base_latency_ms * complexity_mult * fatigue * math.exp(sigma * z)
        if complexity >= _BIG_DECISION and u_tank < level * profile.tank_prob:
            delay += level * profile.tank_extra_ms * (0.5 + u_size)
        return delay


@dataclass(frozen=True, slots=True)
class TimingReport:
    """Summary of a delay sample (ms)."""

    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    cv: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict for printing/CSV."""
        return {"n": self.n, "mean": self.mean, "p50": self.p50, "p95": self.p95, "p99": self.p99, "cv": self.cv}


def timing_report(delays: Sequence[float]) -> TimingReport:
    """mean/p50/p95/p99 and coefficient of variation (population std / mean)."""
    if not delays:
        raise ValueError("empty sample")
    arr = np.asarray(delays, dtype=np.float64)
    mean = float(arr.mean())
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    return TimingReport(n=arr.size, mean=mean, p50=p50, p95=p95, p99=p99,
                        cv=float(arr.std()) / mean if mean > 0 else 0.0)
