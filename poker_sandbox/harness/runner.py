"""Run profiles x detectors x seeds on the offline mock, with a disk cache and failure isolation.

Parallelism: ``ProcessPoolExecutor`` with the ``spawn`` context. Every table runs in a fresh
interpreter state (no shared MockAdapter/RNG/module state), simulation is CPU-bound (GIL), and
each result is a pure function of (config, settings), so the output does not depend on the
number of workers. Detectors run in the main process: a user's detector never needs pickling.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np

from adversarial.oracle import DetectorOracle
from adversarial.runner import RunOutput, RunSettings, run_config
from adversarial.space import AdversarialConfig
from collusion.metrics import collusion_metrics
from harness.config import DetectorSpec, ExperimentConfig, ResolvedProfile, checkpoints
from harness.detectors import build_detector
from harness.env import BATCH_SIZE, PACKAGE_ROOT, __version__, git_hash
from strategy.stats import collect

__all__ = [
    "CacheKeyParts",
    "ExperimentResult",
    "RunCache",
    "RunFn",
    "RunRecord",
    "cache_key",
    "jsonable",
    "run_experiment",
    "run_metrics",
]

RunFn = Callable[[AdversarialConfig, RunSettings], RunOutput]
Status = Literal["ok", "failed"]
DEFAULT_ROOT = PACKAGE_ROOT / "logs" / "harness"


def jsonable(value: Any) -> Any:
    """NaN/inf → ``None`` recursively (strict JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


# ------------------------------------------------------------------ cache


@dataclass(frozen=True, slots=True)
class CacheKeyParts:
    """Everything a cached result depends on; changing any field changes the key."""

    profile_id: str
    detector_id: str
    seed: int
    n_hands: int
    config_hash: str
    equity_samples: int
    batch_size: int
    harness_version: str
    git_hash: str
    n_seats: int = 6
    checkpoints: tuple[int, ...] = ()


def cache_key(parts: CacheKeyParts) -> str:
    """sha256 of the canonical JSON of ``parts``."""
    raw = json.dumps(asdict(parts), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass(slots=True)
class RunRecord:
    """One (profile, detector, seed) result as written to ``runs.jsonl``."""

    profile_id: str
    label: str
    twin_of: str | None
    detector_id: str
    detector_name: str
    detector_version: str
    seed: int
    n_hands: int
    config_hash: str
    cache_key: str
    config: dict[str, Any]
    status: Status
    error: str | None = None
    score: dict[str, Any] | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    wtsd: float | None = None
    prefix: list[list[Any]] = field(default_factory=list)  # [[n_hands, is_bot, confidence], ...]
    cached: bool = False
    timing: dict[str, Any] = field(default_factory=dict)  # volatile: ts, duration_s

    @property
    def ok(self) -> bool:
        """Status is ``ok``."""
        return self.status == "ok"

    def flagged(self, threshold: float) -> bool:
        """Detector says bot with confidence ≥ ``threshold``."""
        return bool(self.ok and self.score and self.score["is_bot"] and self.score["confidence"] >= threshold)

    def bot_belief(self) -> float:
        """Signed belief in bot (NaN when failed)."""
        if not self.ok or not self.score:
            return math.nan
        return float(self.score["confidence"]) * (1 if self.score["is_bot"] else -1)

    def to_dict(self) -> dict[str, Any]:
        """Strict-JSON dict."""
        out: dict[str, Any] = jsonable({f: getattr(self, f) for f in self.__dataclass_fields__})
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Inverse of :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RunCache:
    """``results.jsonl`` (ok records only) + ``failures.jsonl`` (keys of failed runs) in ``directory``."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.results_path = self.directory / "results.jsonl"
        self.failures_path = self.directory / "failures.jsonl"
        self.results: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, dict[str, Any]] = {}
        for path, target in ((self.results_path, self.results), (self.failures_path, self.failures)):
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    target[row["cache_key"]] = row
        for key in self.results:  # a later success clears an old failure
            self.failures.pop(key, None)

    def put(self, record: RunRecord) -> None:
        """Store an ok record; a failed one only goes to the failures list (never reused as a result)."""
        row = record.to_dict()
        if record.ok:
            self.results[record.cache_key] = row
            path = self.results_path
        else:
            self.failures[record.cache_key] = row
            path = self.failures_path
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------- per run


def run_metrics(out: RunOutput, names: Sequence[str]) -> dict[str, float]:
    """Requested M4 collusion metrics and own stats of the target seats (WTSD always included)."""
    targets = [int(s) for s in out.agent_meta["target_seats"]]
    bb = int(out.agent_meta.get("big_blind", 2))
    stats = collect(out.hand_history, {s: "t" for s in targets}, bb)["t"].ratios()
    delays = [float(r["planned_delay_ms"]) for r in out.behavior_log]
    own = {
        "vpip": stats["vpip"], "pfr": stats["pfr"], "af": stats["af"], "wtsd": stats["wtsd"],
        "bb100": stats["bb100"],
        "timing_cv": float(np.std(delays) / np.mean(delays)) if delays and np.mean(delays) > 0 else 0.0,
        "error_share": sum(bool(r["error_applied"]) for r in out.behavior_log) / max(len(out.behavior_log), 1),
        "adjustments": float(out.adjustments),
    }
    values = {**collusion_metrics(out.hand_history, targets, bb), **own}
    return {name: float(values[name]) for name in dict.fromkeys([*names, "wtsd"])}


def score_prefixes(oracle: DetectorOracle, out: RunOutput, cps: Sequence[int]) -> list[list[Any]]:
    """Score growing prefixes of the run: ``[[n, is_bot, confidence, score_dict], ...]``."""
    hand_ids = [r.hand_id for r in out.hand_history]
    rows: list[list[Any]] = []
    for n in cps:
        ids = set(hand_ids[:n])
        score = oracle.score(list(out.hand_history[:n]), {**out.agent_meta, "n_hands": n},
                             [dict(b) for b in out.behavior_log if b.get("hand_id") in ids])
        rows.append([n, score.is_bot, score.confidence, score.to_dict()])
    return rows


# ------------------------------------------------------------- execution


@dataclass(frozen=True, slots=True)
class _Job:
    config: dict[str, Any]
    settings: RunSettings
    run_fn: RunFn


def _play(job: _Job) -> tuple[RunOutput | None, str | None, float]:
    """Worker entry point: never raises, the error text is returned instead."""
    started = time.perf_counter()
    try:
        out = job.run_fn(AdversarialConfig.from_dict(job.config), job.settings)
    except Exception as exc:  # one broken run must not stop the experiment
        return None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started
    return out, None, time.perf_counter() - started


@dataclass(slots=True)
class ExperimentResult:
    """Outcome of :func:`run_experiment`."""

    records: list[RunRecord]
    output_dir: Path | None
    tables_planned: int
    tables_run: int = 0
    cached: int = 0
    failed: int = 0
    skipped_failed: int = 0
    report: Any = None
    started_at: str = ""
    ended_at: str = ""


def _pending_record(p: ResolvedProfile, d: DetectorSpec, seed: int, n_hands: int, key: str) -> RunRecord:
    return RunRecord(profile_id=p.name, label=p.label, twin_of=p.twin_of, detector_id=d.detector_id,
                     detector_name=d.name, detector_version=d.version, seed=seed, n_hands=n_hands,
                     config_hash=p.config.config_hash(), cache_key=key, config=p.config.to_dict(), status="failed")


def run_experiment(
    exp: ExperimentConfig,
    *,
    workers: int = 1,
    resume: bool = False,
    retry_failed: bool = False,
    detectors: Sequence[str] = (),
    dry_run: bool = False,
    output_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    run_fn: RunFn = run_config,
    git: str | None = None,
    progress_every: int = 5,
    log: Callable[[str], None] = print,
) -> ExperimentResult:
    """Plan, execute (cache-aware), score, write ``runs.jsonl`` and the reports.

    Args:
        detectors: Keep only detectors whose name or kind is listed (CLI ``--detector``).
        resume: Reuse cached ok results; known failures are skipped unless ``retry_failed``.
        run_fn: Table runner (tests inject counters/failures); must be picklable for ``workers > 1``.
        git: Git hash override (``None`` = ask git, ``"unknown"`` when unavailable).
    """
    from harness.report import write_reports  # local import: report depends on this module's records

    started = datetime.now(UTC)
    specs = [d for d in exp.detectors if not detectors or d.name in detectors or d.kind in detectors]
    if not specs:
        raise ValueError(f"no detector matches {list(detectors)}")
    profiles = exp.resolved_profiles()
    gh = git if git is not None else git_hash()
    cps = tuple(checkpoints(exp.n_hands_per_run))
    settings_of = {seed: RunSettings(n_hands=exp.n_hands_per_run, seed=seed, n_seats=exp.n_seats,
                                     equity_samples=exp.equity_samples) for seed in exp.seeds}
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_ROOT / "cache"
    cache = RunCache(root)

    records: dict[tuple[int, int, int], RunRecord] = {}
    todo: dict[tuple[str, int], list[tuple[tuple[int, int, int], ResolvedProfile, DetectorSpec]]] = {}
    skipped = cached = 0
    for pi, p in enumerate(profiles):
        for di, d in enumerate(specs):
            for si, seed in enumerate(exp.seeds):
                key = cache_key(CacheKeyParts(p.name, d.detector_id, seed, exp.n_hands_per_run, p.config.config_hash(),
                                              exp.equity_samples, BATCH_SIZE, __version__, gh, exp.n_seats, cps))
                idx = (pi, di, si)
                if resume and key in cache.results:
                    rec = RunRecord.from_dict(cache.results[key])
                    rec.cached = True
                    records[idx] = rec
                    cached += 1
                elif resume and key in cache.failures and not retry_failed:
                    rec = RunRecord.from_dict(cache.failures[key])
                    rec.cached = True
                    records[idx] = rec
                    skipped += 1
                else:  # one table per (effective config, seed): auto baselines equal to their profile share it
                    todo.setdefault((p.config.config_hash(), seed), []).append((idx, p, d))
    total = len(todo)
    log(f"[harness] {exp.name}: {len(profiles)} profiles x {len(specs)} detectors x {len(exp.seeds)} seeds = "
        f"{len(profiles) * len(specs) * len(exp.seeds)} runs; tables to play: {total}; cached: {cached}; "
        f"known failures skipped: {skipped}")
    result = ExperimentResult([], None, total, cached=cached, skipped_failed=skipped,
                              started_at=started.isoformat())
    if dry_run:
        return result

    oracles = {d.name: build_detector(d) for d in specs}
    jobs = {k: _Job(group[0][1].config.to_dict(), settings_of[k[1]], run_fn) for k, group in todo.items()}

    def finish(k: tuple[str, int], out: RunOutput | None, error: str | None, duration: float) -> None:
        result.tables_run += 1
        for idx, p, d in todo[k]:
            rec = _pending_record(p, d, k[1], exp.n_hands_per_run, "")
            rec.cache_key = cache_key(CacheKeyParts(p.name, d.detector_id, k[1], exp.n_hands_per_run, k[0],
                                                    exp.equity_samples, BATCH_SIZE, __version__, gh, exp.n_seats,
                                                    cps))
            rec.timing = {"ts": datetime.now(UTC).isoformat(), "duration_s": round(duration, 3)}
            if out is None:
                rec.error = error
            else:
                try:
                    prefix = score_prefixes(oracles[d.name], out, cps)
                    rec.score = prefix[-1][3]
                    rec.prefix = [row[:3] for row in prefix]
                    rec.metrics = jsonable(run_metrics(out, exp.metrics))
                    rec.wtsd = rec.metrics.get("wtsd")
                    rec.status = "ok"
                except Exception as exc:  # a crashing detector fails only its own runs
                    rec.error = f"{type(exc).__name__}: {exc}"
            if not rec.ok:
                result.failed += 1
            cache.put(rec)
            records[idx] = rec
        if result.tables_run % max(progress_every, 1) == 0 or result.tables_run == total:
            log(f"[harness] tables {result.tables_run}/{total} done, failed runs: {result.failed}")

    if workers <= 1 or total <= 1:
        for k, job in jobs.items():
            finish(k, *_play(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_play, job): k for k, job in jobs.items()}
            for fut in as_completed(futures):
                finish(futures[fut], *fut.result())

    result.records = [records[i] for i in sorted(records)]  # fixed order regardless of completion order
    ended = datetime.now(UTC)
    result.ended_at = ended.isoformat()
    out_dir = Path(output_dir) if output_dir is not None else exp.resolve_output_dir(DEFAULT_ROOT, started)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for rec in result.records:
            fh.write(json.dumps(rec.to_dict(), separators=(",", ":")) + "\n")
    result.output_dir = out_dir
    result.report = write_reports(exp, result.records, out_dir, specs=specs, git=gh, started=result.started_at,
                                  ended=result.ended_at)
    return result
