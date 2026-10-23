"""Detector registry: ``DetectorSpec.kind`` → factory returning a :class:`DetectorOracle`.

Only the M6 ``FakeOracle`` stub ships here; the real detector is registered by the user with
:func:`register_detector` and is not part of M7.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adversarial.oracle import DetectorOracle, FakeOracle
from harness.config import DetectorSpec

__all__ = ["DETECTOR_FACTORIES", "DetectorFactory", "build_detector", "register_detector"]

DetectorFactory = Callable[[dict[str, Any]], DetectorOracle]


def _fake(params: dict[str, Any]) -> DetectorOracle:
    return FakeOracle(**params)


DETECTOR_FACTORIES: dict[str, DetectorFactory] = {"fake": _fake}


def register_detector(kind: str, factory: DetectorFactory) -> None:
    """Make ``kind`` usable in :class:`DetectorSpec` (detectors run in the main process: no pickling needed)."""
    DETECTOR_FACTORIES[kind] = factory


def build_detector(spec: DetectorSpec) -> DetectorOracle:
    """Instantiate the detector of ``spec``."""
    try:
        factory = DETECTOR_FACTORIES[spec.kind]
    except KeyError:
        raise ValueError(f"unknown detector kind {spec.kind!r}; registered: {sorted(DETECTOR_FACTORIES)}") from None
    oracle = factory(dict(spec.params))
    if not isinstance(oracle, DetectorOracle):
        raise TypeError(f"factory for {spec.kind!r} did not return a DetectorOracle")
    return oracle
