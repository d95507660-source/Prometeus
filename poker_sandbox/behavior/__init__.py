"""Behaviour layer (M5): human-like timings, errors and tilt over any Strategy, for detector calibration."""

from behavior.humanlike import HumanLikeWrapper, SupportsBluffBonus, SupportsEquity, error_action
from behavior.logging import BehaviorLogger, BehaviorRecord
from behavior.profiles import (
    ERROR_KINDS,
    FAST_HUMAN,
    MAX_ERROR_RATE,
    PRESETS,
    ROBOT,
    SLOW_HUMAN,
    TILTING_HUMAN,
    TIRED_HUMAN,
    ErrorKind,
    HumanLikeProfile,
)
from behavior.timing import DecisionContext, TimingModel, TimingReport, context_from_state, timing_report

__all__ = [
    "ERROR_KINDS", "FAST_HUMAN", "MAX_ERROR_RATE", "PRESETS", "ROBOT", "SLOW_HUMAN", "TILTING_HUMAN",
    "TIRED_HUMAN", "BehaviorLogger", "BehaviorRecord", "DecisionContext", "ErrorKind", "HumanLikeProfile",
    "HumanLikeWrapper", "SupportsBluffBonus", "SupportsEquity", "TimingModel", "TimingReport",
    "context_from_state", "error_action", "timing_report",
]
