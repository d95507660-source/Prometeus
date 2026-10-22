"""Parameters of the heuristic strategy (plain dataclass, validated in ``__post_init__``)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeuristicParams:
    """Knobs of :class:`~strategy.heuristic.HeuristicStrategy`.

    Attributes:
        vpip_scale: Multiplier of the share of hands played voluntarily (calls/limps included).
        pfr_scale: Multiplier of preflop raising ranges (opens, 3-bet/4-bet value).
        aggression: 0..1; shifts thresholds and sizings towards betting/raising.
        bluff_freq: 0..1; probability scale of bluffs (3-bet bluffs, postflop air bets).
        cbet_freq: 0..1; flop continuation-bet frequency as preflop aggressor in a single-raised pot.
        sizing_set: Bet sizes as fractions of the pot; larger ones get likelier with aggression.
        equity_samples: Monte Carlo samples per postflop decision.
        seed: RNG seed; same seed + same states → same actions.
    """

    vpip_scale: float = 1.0
    pfr_scale: float = 1.0
    aggression: float = 0.5
    bluff_freq: float = 0.1
    cbet_freq: float = 0.6
    sizing_set: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0)
    equity_samples: int = 600
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("vpip_scale", "pfr_scale"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("aggression", "bluff_freq", "cbet_freq"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.sizing_set:
            raise ValueError("sizing_set must not be empty")
        if any(not (math.isfinite(s) and s > 0) for s in self.sizing_set):
            raise ValueError("sizing_set values must be positive")
        if self.equity_samples < 50:
            raise ValueError("equity_samples must be >= 50")


TIGHT = HeuristicParams(vpip_scale=0.7, pfr_scale=0.8, aggression=0.4, bluff_freq=0.05, cbet_freq=0.55)
LOOSE = HeuristicParams(vpip_scale=1.7, pfr_scale=1.0, aggression=0.35, bluff_freq=0.1, cbet_freq=0.5)
AGGRESSIVE = HeuristicParams(vpip_scale=1.25, pfr_scale=1.6, aggression=0.85, bluff_freq=0.25, cbet_freq=0.8)
