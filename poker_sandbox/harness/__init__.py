"""Experiment harness (M7): profiles x detectors x seeds, detector quality metrics, reproducible reports."""

from harness.config import (
    KNOWN_METRICS,
    DetectorSpec,
    ExperimentConfig,
    ProfileSpec,
    ResolvedProfile,
    checkpoints,
)
from harness.detectors import DETECTOR_FACTORIES, build_detector, register_detector
from harness.env import BATCH_SIZE, __version__, git_hash, versions
from harness.metrics import (
    Stat,
    Summary,
    compute_summary,
    detection_matrix,
    detection_rate,
    false_positive_rate,
    mean_ci,
    time_to_detect,
)
from harness.report import ReportPaths, build_manifest, write_reports
from harness.runner import CacheKeyParts, ExperimentResult, RunCache, RunRecord, cache_key, run_experiment

__all__ = [
    "BATCH_SIZE", "DETECTOR_FACTORIES", "KNOWN_METRICS", "CacheKeyParts", "DetectorSpec", "ExperimentConfig",
    "ExperimentResult", "ProfileSpec", "ReportPaths", "ResolvedProfile", "RunCache", "RunRecord", "Stat", "Summary",
    "__version__", "build_detector", "build_manifest", "cache_key", "checkpoints", "compute_summary",
    "detection_matrix", "detection_rate", "false_positive_rate", "git_hash", "mean_ci", "register_detector",
    "run_experiment", "time_to_detect", "versions", "write_reports",
]
