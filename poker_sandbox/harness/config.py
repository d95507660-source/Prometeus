"""Experiment configuration (pydantic v2, JSON files).

JSON, not YAML: pydantic v2 parses/validates it natively and no new dependency is needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adversarial.space import AdversarialConfig, ParamSpace
from collusion.metrics import collusion_metrics

__all__ = [
    "COLLUSION_METRICS",
    "KNOWN_METRICS",
    "OWN_METRICS",
    "DetectorSpec",
    "ExperimentConfig",
    "ProfileSpec",
    "ResolvedProfile",
    "checkpoints",
]

COLLUSION_METRICS: tuple[str, ...] = tuple(collusion_metrics([], (0, 1), 2))  # names straight from M4
OWN_METRICS: tuple[str, ...] = ("vpip", "pfr", "af", "wtsd", "bb100", "timing_cv", "error_share", "adjustments")
KNOWN_METRICS: tuple[str, ...] = COLLUSION_METRICS + OWN_METRICS
_SPACE = ParamSpace()


class ProfileSpec(BaseModel):
    """A bot profile: AdversarialConfig overrides on top of its defaults.

    ``label="baseline"`` puts the profile into the negative class used for FPR.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.\-]+$")
    config: dict[str, Any] = Field(default_factory=dict)
    label: Literal["bot", "baseline"] = "bot"

    @field_validator("config")
    @classmethod
    def _valid_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        _SPACE.validate(AdversarialConfig.from_dict(value))
        return value

    def adversarial(self) -> AdversarialConfig:
        """The full :class:`AdversarialConfig`."""
        return AdversarialConfig.from_dict(self.config)


class DetectorSpec(BaseModel):
    """A detector: ``kind`` selects a factory (``"fake"`` = M6 FakeOracle stub), ``params`` go to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.\-]+$")
    kind: str = "fake"
    version: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def detector_id(self) -> str:
        """``name@version#params-hash``: any change of version or params is a different detector."""
        digest = hashlib.sha256(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.name}@{self.version}#{digest}"


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """A profile after baseline expansion."""

    name: str
    config: AdversarialConfig
    label: Literal["bot", "baseline"]
    twin_of: str | None = None  # for auto baselines: the profile they mirror


def checkpoints(n_hands: int) -> list[int]:
    """Prefix sizes for time-to-detect: 1-2-5 series from 10 (10, 20, 50, 100, 200, 500, …) plus ``n_hands``."""
    out: list[int] = []
    base = 10
    while base <= n_hands:
        out += [m * base for m in (1, 2, 5) if m * base <= n_hands]
        base *= 10
    if not out or out[-1] != n_hands:
        out.append(n_hands)
    return out


class ExperimentConfig(BaseModel):
    """Profiles x detectors x seeds, plus run settings and reporting knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.\-]+$")
    n_hands_per_run: int = Field(ge=100)
    seeds: list[int] = Field(min_length=1)
    profiles: list[ProfileSpec] = Field(min_length=1)
    detectors: list[DetectorSpec] = Field(min_length=1)
    baseline: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["sync_rate", "yield_rate", "sizing_spearman", "vpip",
                                                         "pfr", "wtsd", "timing_cv"])
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    equity_samples: int = Field(default=200, ge=50)
    n_seats: int = Field(default=6, ge=2, le=6)
    output_dir: Path | None = None
    notes: str = ""

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("seeds must be unique")
        return value

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: list[str]) -> list[str]:
        if unknown := sorted(set(value) - set(KNOWN_METRICS)):
            raise ValueError(f"unknown metrics {unknown}; known: {list(KNOWN_METRICS)}")
        return value

    @model_validator(mode="after")
    def _unique_names(self) -> ExperimentConfig:
        names = [p.name for p in self.profiles]
        if len(set(names)) != len(names):
            raise ValueError("profile names must be unique")
        if any(n.endswith("__baseline") for n in names):
            raise ValueError("the suffix '__baseline' is reserved for auto baselines")
        dets = [d.name for d in self.detectors]
        if len(set(dets)) != len(dets):
            raise ValueError("detector names must be unique")
        for p in self.profiles:
            if p.adversarial().group_size > self.n_seats - 1:
                raise ValueError(f"profile {p.name}: group must leave a field seat at {self.n_seats} seats")
        return self

    # ----------------------------------------------------------------- io

    @classmethod
    def from_file(cls, path: Path | str) -> ExperimentConfig:
        """Load and validate a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_file(self, path: Path | str) -> Path:
        """Write as indented JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    def config_hash(self) -> str:
        """sha256 of the canonical config without ``output_dir`` (where results go does not change them)."""
        data = self.model_dump(mode="json", exclude={"output_dir"})
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def resolve_output_dir(self, root: Path, now: datetime) -> Path:
        """``output_dir`` or ``<root>/<name>_<YYYYmmdd-HHMMSS>``."""
        return self.output_dir or root / f"{self.name}_{now.strftime('%Y%m%d-%H%M%S')}"

    def resolved_profiles(self) -> list[ResolvedProfile]:
        """Profiles in order; with ``baseline`` each bot gets a ``<name>__baseline`` twin right after it.

        The twin keeps the strategy knobs but drops coordination and human-like noise.
        """
        out = []
        for p in self.profiles:
            cfg = p.adversarial()
            out.append(ResolvedProfile(name=p.name, config=cfg, label=p.label))
            if self.baseline and p.label == "bot":
                twin = AdversarialConfig.from_dict({**cfg.to_dict(), "collusion": False, "group_size": 1,
                                                    "humanlike_level": 0.0})
                out.append(ResolvedProfile(name=f"{p.name}__baseline", config=twin, label="baseline",
                                           twin_of=p.name))
        return out
