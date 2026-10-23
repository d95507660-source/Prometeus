"""Parameter space of synthetic bot configurations for fuzzing a detector.

One :class:`AdversarialConfig` = strategy knobs (M3) + human-like noise (M5) + coordination (M4).
:class:`ParamSpace` knows the bounds and samples, mutates, crosses over and validates configs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from behavior.profiles import HumanLikeProfile
from collusion.policy import CoordinationParams
from strategy.params import HeuristicParams

__all__ = [
    "DEFAULT_AXES",
    "SIZING_SETS",
    "AdversarialConfig",
    "Axis",
    "ChoiceAxis",
    "FloatAxis",
    "IntAxis",
    "ParamSpace",
]

SIZING_SETS: dict[str, tuple[float, ...]] = {
    "small": (0.25, 0.33, 0.5),
    "default": (0.33, 0.5, 0.75, 1.0),
    "big": (0.5, 0.75, 1.0, 1.5),
    "polar": (0.33, 1.25),
}
_DIGITS = 4  # floats are rounded so configs print, hash and round-trip cleanly


@dataclass(frozen=True, slots=True)
class AdversarialConfig:
    """One point of the space. Bounds are checked by :meth:`ParamSpace.validate`; here only consistency.

    ``collusion`` and ``group_size`` must agree: off ⇔ ``group_size == 1``.
    """

    # strategy (M3)
    vpip_scale: float = 1.0
    pfr_scale: float = 1.0
    aggression: float = 0.5
    bluff_freq: float = 0.1
    cbet_freq: float = 0.6
    sizing: str = "default"
    # human-like noise (M5)
    humanlike_level: float = 0.0
    error_rate: float = 0.04
    base_latency_ms: float = 1200.0
    latency_sigma: float = 0.45
    tilt_threshold: float = 30.0
    # coordination (M4); exploration defaults are the strong ones (strength 1, interval 1)
    collusion: bool = False
    coordination_strength: float = 1.0
    signal_interval: int = 1
    group_size: int = 1

    def __post_init__(self) -> None:
        if self.sizing not in SIZING_SETS:
            raise ValueError(f"unknown sizing {self.sizing!r}; expected one of {sorted(SIZING_SETS)}")
        if not isinstance(self.collusion, bool):
            raise ValueError("collusion must be a bool")
        for name in ("signal_interval", "group_size"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an int")
        if self.collusion != (self.group_size >= 2):
            raise ValueError("collusion=True needs group_size >= 2, collusion=False needs group_size == 1")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{f.name} must be finite")

    # ------------------------------------------------------------ conversion

    def heuristic_params(self, seed: int | None, equity_samples: int = 600) -> HeuristicParams:
        """M3 parameters of the bot."""
        return HeuristicParams(vpip_scale=self.vpip_scale, pfr_scale=self.pfr_scale, aggression=self.aggression,
                               bluff_freq=self.bluff_freq, cbet_freq=self.cbet_freq,
                               sizing_set=SIZING_SETS[self.sizing], equity_samples=equity_samples, seed=seed)

    def humanlike_profile(self, session_hands: int) -> HumanLikeProfile:
        """M5 profile; fields outside the space keep HumanLikeProfile defaults."""
        return HumanLikeProfile(humanlike_level=self.humanlike_level, error_rate=self.error_rate,
                                base_latency_ms=self.base_latency_ms, latency_sigma=self.latency_sigma,
                                tilt_threshold=self.tilt_threshold, session_hands=session_hands)

    def coordination_params(self, seed: int | None) -> CoordinationParams:
        """M4 parameters (only meaningful when ``collusion``)."""
        return CoordinationParams(coordination_strength=self.coordination_strength,
                                  signal_interval=self.signal_interval, seed=seed)

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdversarialConfig:
        """Inverse of :meth:`to_dict`; unknown keys are rejected."""
        names = {f.name for f in fields(cls)}
        if unknown := set(data) - names:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs = dict(data)
        for f in fields(cls):  # JSON has no int/float distinction for 1 vs 1.0
            if f.name in kwargs and f.type == "float" and isinstance(kwargs[f.name], int) \
                    and not isinstance(kwargs[f.name], bool):
                kwargs[f.name] = float(kwargs[f.name])
        return cls(**kwargs)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> AdversarialConfig:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def effective_dict(self) -> dict[str, Any]:
        """Fields that actually change play: coordination knobs vanish without collusion, noise knobs at level 0."""
        d = self.to_dict()
        if not self.collusion:
            for name in ("coordination_strength", "signal_interval"):
                d.pop(name)
        if self.humanlike_level == 0.0:  # M5 at level 0: constant base latency, no errors/tilt
            for name in ("error_rate", "latency_sigma", "tilt_threshold"):
                d.pop(name)
        return d

    def config_hash(self) -> str:
        """Short sha256 of :meth:`effective_dict`; equal for configs that play identically."""
        canonical = json.dumps(self.effective_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# --------------------------------------------------------------- axes


@dataclass(frozen=True, slots=True)
class FloatAxis:
    """Continuous field in ``[lo, hi]``."""

    name: str
    lo: float
    hi: float
    group: str


@dataclass(frozen=True, slots=True)
class IntAxis:
    """Integer field in ``[lo, hi]``."""

    name: str
    lo: int
    hi: int
    group: str


@dataclass(frozen=True, slots=True)
class ChoiceAxis:
    """Categorical field."""

    name: str
    choices: tuple[str | bool, ...]
    group: str


Axis = FloatAxis | IntAxis | ChoiceAxis

DEFAULT_AXES: tuple[Axis, ...] = (
    FloatAxis("vpip_scale", 0.5, 2.0, "heuristic"),
    FloatAxis("pfr_scale", 0.5, 2.0, "heuristic"),
    FloatAxis("aggression", 0.0, 1.0, "heuristic"),
    FloatAxis("bluff_freq", 0.0, 0.4, "heuristic"),
    FloatAxis("cbet_freq", 0.2, 1.0, "heuristic"),
    ChoiceAxis("sizing", tuple(SIZING_SETS), "heuristic"),
    FloatAxis("humanlike_level", 0.0, 1.0, "humanlike"),
    FloatAxis("error_rate", 0.0, 0.15, "humanlike"),
    FloatAxis("base_latency_ms", 300.0, 4000.0, "humanlike"),
    FloatAxis("latency_sigma", 0.0, 1.0, "humanlike"),
    FloatAxis("tilt_threshold", 5.0, 100.0, "humanlike"),
    ChoiceAxis("collusion", (False, True), "collusion"),
    FloatAxis("coordination_strength", 0.0, 1.0, "collusion"),
    IntAxis("signal_interval", 1, 5, "collusion"),
    IntAxis("group_size", 1, 3, "collusion"),
)


def _repair(d: dict[str, Any]) -> dict[str, Any]:
    """Make collusion and group_size agree (the group decides only when collusion is on)."""
    if not d["collusion"]:
        d["group_size"] = 1
    elif d["group_size"] < 2:
        d["group_size"] = 2
    return d


class ParamSpace:
    """Bounds of every field plus sampling/mutation/crossover; all randomness from the passed ``rng``."""

    def __init__(self, axes: tuple[Axis, ...] = DEFAULT_AXES) -> None:
        names = [a.name for a in axes]
        expected = {f.name for f in fields(AdversarialConfig)}
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("axes must cover every AdversarialConfig field exactly once")
        self.axes = axes
        self._by_name = {a.name: a for a in axes}

    def axis(self, name: str) -> Axis:
        """Axis of field ``name``."""
        return self._by_name[name]

    @property
    def numeric_fields(self) -> tuple[str, ...]:
        """Fields usable for correlations (floats, ints, bools)."""
        return tuple(a.name for a in self.axes
                     if not isinstance(a, ChoiceAxis) or all(isinstance(c, bool) for c in a.choices))

    def validate(self, config: AdversarialConfig) -> None:
        """Raise ``ValueError`` if any field is outside its axis."""
        for a in self.axes:
            value = getattr(config, a.name)
            if isinstance(a, ChoiceAxis):
                if value not in a.choices:
                    raise ValueError(f"{a.name}={value!r} not in {a.choices}")
            elif not a.lo <= value <= a.hi:
                raise ValueError(f"{a.name}={value} outside [{a.lo}, {a.hi}]")

    def midpoint(self, **overrides: Any) -> AdversarialConfig:
        """Middle of every continuous axis, first choice of categorical ones (``sizing="default"``)."""
        d: dict[str, Any] = {}
        for a in self.axes:
            if isinstance(a, FloatAxis):
                d[a.name] = round((a.lo + a.hi) / 2, _DIGITS)
            elif isinstance(a, IntAxis):
                d[a.name] = (a.lo + a.hi) // 2
            else:
                d[a.name] = a.choices[0]
        d["sizing"] = "default" if "default" in self._choices("sizing") else d["sizing"]
        d.update(overrides)
        if "group_size" not in overrides:
            _repair(d)
        config = AdversarialConfig.from_dict(d)
        self.validate(config)
        return config

    def sample_uniform(self, rng: random.Random) -> AdversarialConfig:
        """Uniform over every axis (group_size drawn from 2..hi when collusion is on)."""
        d: dict[str, Any] = {}
        for a in self.axes:  # fixed axis order → same draws for the same seed
            if isinstance(a, FloatAxis):
                d[a.name] = round(rng.uniform(a.lo, a.hi), _DIGITS)
            elif isinstance(a, IntAxis):
                d[a.name] = rng.randint(a.lo, a.hi)
            else:
                d[a.name] = rng.choice(a.choices)
        if d["collusion"]:
            gs = self._by_name["group_size"]
            assert isinstance(gs, IntAxis)
            d["group_size"] = rng.randint(max(2, gs.lo), gs.hi)
        return AdversarialConfig.from_dict(_repair(d))

    def mutate(self, config: AdversarialConfig, rng: random.Random, strength: float = 0.2,
               jump_prob: float = 0.1) -> AdversarialConfig:
        """Gaussian noise of ``strength x range`` on numeric axes; with ``jump_prob`` a uniform re-draw."""
        if not 0 < strength <= 1 or not 0 <= jump_prob <= 1:
            raise ValueError("strength must be in (0, 1], jump_prob in [0, 1]")
        d = config.to_dict()
        for a in self.axes:
            jump = rng.random() < jump_prob
            noise = rng.normalvariate(0.0, 1.0)  # always drawn: keeps the stream aligned
            if isinstance(a, ChoiceAxis):
                if jump or abs(noise) * strength > 0.5:  # categorical flips are rarer than numeric nudges
                    d[a.name] = rng.choice(a.choices)
                continue
            span = a.hi - a.lo
            value = rng.uniform(a.lo, a.hi) if jump else d[a.name] + noise * strength * span
            value = min(a.hi, max(a.lo, value))
            d[a.name] = round(value) if isinstance(a, IntAxis) else round(value, _DIGITS)
        return AdversarialConfig.from_dict(_repair(d))

    def crossover(self, a: AdversarialConfig, b: AdversarialConfig, rng: random.Random) -> AdversarialConfig:
        """Uniform crossover; the three collusion fields travel together so a group stays coherent."""
        da, db = a.to_dict(), b.to_dict()
        child: dict[str, Any] = {}
        collusion_from_a = rng.random() < 0.5
        for ax in self.axes:
            if ax.group == "collusion":
                child[ax.name] = (da if collusion_from_a else db)[ax.name]
            else:
                child[ax.name] = (da if rng.random() < 0.5 else db)[ax.name]
        return AdversarialConfig.from_dict(_repair(child))

    def with_fields(self, config: AdversarialConfig, **changes: Any) -> AdversarialConfig:
        """``dataclasses.replace`` + validation."""
        new = replace(config, **changes)
        self.validate(new)
        return new

    def _choices(self, name: str) -> tuple[str | bool, ...]:
        a = self._by_name[name]
        return a.choices if isinstance(a, ChoiceAxis) else ()
