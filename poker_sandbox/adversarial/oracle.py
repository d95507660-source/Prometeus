"""Interface to the detector under test (``DetectorOracle``) and a toy stand-in for self-tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from adapter.models import HandResult
from collusion.metrics import collusion_metrics
from strategy.stats import collect

__all__ = ["DetectorOracle", "DetectorScore", "FakeOracle"]


@dataclass(frozen=True, slots=True)
class DetectorScore:
    """Verdict of a detector.

    Attributes:
        is_bot: Detector says "bot".
        confidence: 0..1, confidence in ``is_bot`` (not in "bot").
        reason: Short human-readable explanation.
        features: Feature values the detector used (for analysing what drives it).
    """

    is_bot: bool
    confidence: float
    reason: str = ""
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def bot_belief(self) -> float:
        """Signed belief in "bot": +confidence if ``is_bot`` else -confidence (-1..1)."""
        return self.confidence if self.is_bot else -self.confidence

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (NaN features become ``None``)."""
        return {"is_bot": self.is_bot, "confidence": self.confidence, "reason": self.reason,
                "features": {k: (None if isinstance(v, float) and math.isnan(v) else v)
                             for k, v in self.features.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorScore:
        """Inverse of :meth:`to_dict`."""
        feats = {k: math.nan if v is None else float(v) for k, v in data.get("features", {}).items()}
        return cls(bool(data["is_bot"]), float(data["confidence"]), str(data.get("reason", "")), feats)


@runtime_checkable
class DetectorOracle(Protocol):
    """What M6 needs from a detector. Plug the real one in through this Protocol."""

    def score(self, hand_history: list[HandResult], agent_meta: dict[str, Any],
              behavior_log: list[dict[str, Any]]) -> DetectorScore:
        """Judge the seats in ``agent_meta["target_seats"]`` from their hands and behaviour log."""
        ...


def _clip(x: float) -> float:
    return 0.0 if math.isnan(x) else min(1.0, max(0.0, x))


class FakeOracle:
    """Toy detector — THIS IS A STUB FOR M6 SELF-TESTS ONLY.

    The real detector is plugged in through the same :class:`DetectorOracle` Protocol and is
    not part of M6. Features: timing evenness (CV of planned delays), low VPIP, high WTSD
    (M3 bots sit at 50-60 %, regulars at 25-30 %) and, for a group, the M4 yield rate
    (folds to a partner's aggression). Bot score = weighted mean of the clipped components.
    """

    def __init__(self, *, cv_human: float = 0.3, vpip_low: float = 18.0, wtsd_human: float = 32.0,
                 wtsd_bot: float = 55.0, yield_human: float = 0.5, threshold: float = 0.5,
                 weights: Mapping[str, float] | None = None) -> None:
        self.cv_human = cv_human
        self.vpip_low = vpip_low
        self.wtsd_human = wtsd_human
        self.wtsd_bot = wtsd_bot
        self.yield_human = yield_human
        self.threshold = threshold
        self.weights = dict(weights or {"timing": 0.45, "vpip": 0.15, "wtsd": 0.25, "collusion": 0.15})

    def score(self, hand_history: list[HandResult], agent_meta: dict[str, Any],
              behavior_log: list[dict[str, Any]]) -> DetectorScore:
        """Deterministic function of its inputs."""
        targets = [int(s) for s in agent_meta["target_seats"]]
        bb = int(agent_meta.get("big_blind", 2))
        delays = [float(r["planned_delay_ms"]) for r in behavior_log if r.get("seat") in targets]
        cv = float(np.std(delays) / np.mean(delays)) if delays and np.mean(delays) > 0 else 0.0
        stats = collect(hand_history, {s: "t" for s in targets}, bb)["t"].ratios()
        yield_rate = math.nan
        if len(targets) >= 2:
            yield_rate = collusion_metrics(hand_history, targets, bb)["yield_rate"]
        parts = {
            "timing": _clip((self.cv_human - cv) / self.cv_human),
            "vpip": _clip((self.vpip_low - stats["vpip"]) / (self.vpip_low / 2)),
            "wtsd": _clip((stats["wtsd"] - self.wtsd_human) / (self.wtsd_bot - self.wtsd_human)),
            "collusion": _clip((yield_rate - self.yield_human) / (1 - self.yield_human)),
        }
        active = {k: w for k, w in self.weights.items() if k != "collusion" or len(targets) >= 2}
        p = sum(parts[k] * w for k, w in active.items()) / sum(active.values())
        is_bot = p >= self.threshold
        top: Sequence[str] = sorted(active, key=lambda k: -parts[k] * active[k])[:2]
        reason = ", ".join(f"{k}={parts[k]:.2f}" for k in top)
        features = {"timing_cv": cv, "vpip": stats["vpip"], "pfr": stats["pfr"], "wtsd": stats["wtsd"],
                    "yield_rate": yield_rate, "bot_score": p, **{f"s_{k}": v for k, v in parts.items()}}
        return DetectorScore(is_bot, p if is_bot else 1 - p, reason, features)
