"""Behaviour profiles: how "human" a wrapped strategy looks in timings, errors and tilt.

Every effect is scaled by ``humanlike_level``: 0 is a perfectly even robot (constant
latency, no errors, no tilt), 1 applies the profile in full.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

__all__ = [
    "ERROR_KINDS",
    "FAST_HUMAN",
    "MAX_ERROR_RATE",
    "PRESETS",
    "ROBOT",
    "SLOW_HUMAN",
    "TILTING_HUMAN",
    "TIRED_HUMAN",
    "ErrorKind",
    "HumanLikeProfile",
]

ErrorKind = Literal["random_fold", "random_call", "min_raise"]
ERROR_KINDS: tuple[ErrorKind, ...] = get_args(ErrorKind)
MAX_ERROR_RATE = 0.15


@dataclass(frozen=True, slots=True)
class HumanLikeProfile:
    """Knobs of :class:`~behavior.humanlike.HumanLikeWrapper` and :class:`~behavior.timing.TimingModel`.

    Attributes:
        base_latency_ms: Median planned delay of a routine decision.
        latency_sigma: Sigma of the lognormal noise (at level 1).
        complexity_factor: Delay multiplier for the hardest decisions (river, big bets, close spots).
        fatigue_drift: Growth of mean delay and of its spread at the end of the session (fraction).
        tilt_threshold: Big blinds lost over consecutive losing hands that put the player on tilt.
        tilt_extra_variance: Extra relative sigma of delays while on tilt.
        tilt_bluff_bonus: Bluff bonus pushed to a delegate that exposes ``bluff_bonus``.
        tilt_error_mult: Error-rate multiplier while on tilt.
        tank_prob: Probability of tanking before a big decision.
        tank_extra_ms: Mean extra delay of a tank.
        error_rate: Probability of a random "human" error per decision (≤ ``MAX_ERROR_RATE``).
        error_kinds: Kinds to draw errors from; repeating a kind raises its weight.
        session_hands: Session length used to place a hand on the fatigue curve.
        humanlike_level: 0..1, global multiplier of every effect above.
    """

    base_latency_ms: float = 1200.0
    latency_sigma: float = 0.45
    complexity_factor: float = 1.8
    fatigue_drift: float = 0.3
    tilt_threshold: float = 30.0
    tilt_extra_variance: float = 0.5
    tilt_bluff_bonus: float = 0.1
    tilt_error_mult: float = 2.0
    tank_prob: float = 0.15
    tank_extra_ms: float = 4000.0
    error_rate: float = 0.04
    error_kinds: tuple[ErrorKind, ...] = ERROR_KINDS
    session_hands: int = 1000
    humanlike_level: float = 1.0

    def __post_init__(self) -> None:
        for name in ("base_latency_ms", "tilt_threshold"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("latency_sigma", "fatigue_drift", "tilt_extra_variance", "tilt_bluff_bonus", "tank_extra_ms"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a non-negative finite number")
        if not (math.isfinite(self.complexity_factor) and self.complexity_factor >= 1.0):
            raise ValueError("complexity_factor must be >= 1")
        if not (math.isfinite(self.tilt_error_mult) and self.tilt_error_mult >= 1.0):
            raise ValueError("tilt_error_mult must be >= 1")
        for name in ("tank_prob", "humanlike_level"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.error_rate <= MAX_ERROR_RATE:
            raise ValueError(f"error_rate must be in [0, {MAX_ERROR_RATE}]")
        if self.error_rate > 0 and not self.error_kinds:
            raise ValueError("error_kinds must not be empty when error_rate > 0")
        if unknown := set(self.error_kinds) - set(ERROR_KINDS):
            raise ValueError(f"unknown error kinds: {sorted(unknown)}")
        if self.session_hands < 1:
            raise ValueError("session_hands must be >= 1")


ROBOT = HumanLikeProfile(humanlike_level=0.0)
FAST_HUMAN = HumanLikeProfile(base_latency_ms=700.0, latency_sigma=0.35, complexity_factor=1.5, tank_prob=0.08,
                              tank_extra_ms=2500.0, error_rate=0.03)
SLOW_HUMAN = HumanLikeProfile(base_latency_ms=2500.0, latency_sigma=0.5, complexity_factor=2.2, tank_prob=0.25,
                              tank_extra_ms=7000.0, error_rate=0.02)
# Weighted towards random_fold: folding more often before showdown also pulls WTSD down (cheap fix for M3's ~55%).
TILTING_HUMAN = HumanLikeProfile(tilt_threshold=15.0, tilt_extra_variance=0.8, tilt_bluff_bonus=0.2,
                                 tilt_error_mult=2.5, error_rate=0.05,
                                 error_kinds=("random_fold", "random_fold", "random_call", "min_raise"))
TIRED_HUMAN = HumanLikeProfile(base_latency_ms=1500.0, fatigue_drift=1.0, latency_sigma=0.5, error_rate=0.06)

PRESETS: dict[str, HumanLikeProfile] = {
    "robot": ROBOT, "fast": FAST_HUMAN, "slow": SLOW_HUMAN, "tilting": TILTING_HUMAN, "tired": TIRED_HUMAN,
}
