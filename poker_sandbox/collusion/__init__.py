"""Coordinated play of several bots (M4): signal protocol, channel, policy, runner, metrics.

Synthetic sandbox only: signals carry coarse strength/intent, never cards; every
message and every adjusted action is logged to JSONL for detector calibration.
"""

from collusion.channel import InProcessChannel, JsonlSink, SignalChannel, make_channel
from collusion.metrics import collusion_metrics, compare_runs, write_csv
from collusion.policy import CoordinationParams, CoordinationPolicy, PartnerSignal
from collusion.protocol import Bucket, Line, Message, SignalIntent, SignalStrength, decode, encode, strength_signal
from collusion.runner import Experiment, ScenarioConfig, ScenarioResult, legality_error, run_experiment, run_scenario

__all__ = [
    "Bucket", "CoordinationParams", "CoordinationPolicy", "Experiment", "InProcessChannel", "JsonlSink", "Line",
    "Message", "PartnerSignal", "ScenarioConfig", "ScenarioResult", "SignalChannel", "SignalIntent",
    "SignalStrength", "collusion_metrics", "compare_runs", "decode", "encode", "legality_error", "make_channel",
    "run_experiment", "run_scenario", "strength_signal", "write_csv",
]
