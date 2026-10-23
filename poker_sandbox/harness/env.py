"""Version and environment facts that make results reproducible (and invalidate caches)."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pydantic

from pokermath.equity import BATCH_SIZE

__all__ = ["BATCH_SIZE", "PACKAGE_ROOT", "__version__", "git_hash", "package_version", "versions"]

__version__ = "0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def git_hash(root: Path = PACKAGE_ROOT) -> str:
    """``git rev-parse HEAD`` of the repository holding ``root`` (local command, no network) or ``"unknown"``."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def package_version() -> str:
    """Version of the sandbox package (pokermath has no own ``__version__``): metadata, else pyproject."""
    try:
        return importlib.metadata.version("poker-sandbox-adapter")
    except importlib.metadata.PackageNotFoundError:
        try:
            data = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
            return str(data["project"]["version"])
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            return "unknown"


def versions() -> dict[str, str]:
    """python, numpy, pydantic, pokermath (= sandbox package) and harness versions."""
    return {"python": platform.python_version(), "numpy": np.__version__, "pydantic": pydantic.VERSION,
            "pokermath": package_version(), "harness": __version__}
