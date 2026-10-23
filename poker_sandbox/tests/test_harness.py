"""Tests for the M7 experiment harness (offline mock only; FakeOracle stands in for the real detector)."""

from __future__ import annotations

import csv
import json
import math
import socket
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from adversarial.runner import RunOutput, RunSettings, run_config
from adversarial.space import AdversarialConfig
from harness import (
    BATCH_SIZE,
    CacheKeyParts,
    ExperimentConfig,
    RunCache,
    RunRecord,
    __version__,
    cache_key,
    checkpoints,
    compute_summary,
    detection_matrix,
    detection_rate,
    false_positive_rate,
    mean_ci,
    run_experiment,
    time_to_detect,
)
from harness.cli import main as cli_main

DEMO = Path(__file__).resolve().parents[1] / "examples" / "exp_demo.json"


def small_exp(**over: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "t",
        "n_hands_per_run": 100,
        "seeds": [1, 2],
        "profiles": [{"name": "robot", "config": {}},
                     {"name": "pair", "config": {"collusion": True, "group_size": 2, "humanlike_level": 0.5}}],
        "detectors": [{"name": "lo", "version": "v1", "params": {"threshold": 0.35}},
                      {"name": "hi", "version": "v1", "params": {"threshold": 0.65}}],
        "baseline": False,
        "equity_samples": 50,
    }
    data.update(over)
    return ExperimentConfig.model_validate(data)


class Counting:
    """Picklable-free counting run function (workers=1)."""

    def __init__(self, fail: Any = None) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, config: AdversarialConfig, settings: RunSettings) -> RunOutput:
        self.calls += 1
        if self.fail is not None and self.fail(config, settings):
            raise RuntimeError("injected failure")
        return run_config(config, settings)


def run(tmp: Path, exp: ExperimentConfig, out: str = "out", **kw: Any) -> Any:
    kw.setdefault("cache_dir", tmp / "cache")
    kw.setdefault("git", "deadbeef")
    return run_experiment(exp, output_dir=tmp / out, log=lambda s: None, **kw)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


# ---------------------------------------------------------------- config


def test_config_validation_and_roundtrip(tmp_path: Path) -> None:
    exp = small_exp()
    bad: dict[str, Any]
    for bad in ({"seeds": []}, {"seeds": [1, 1]}, {"n_hands_per_run": 99}, {"metrics": ["nope"]},
                {"profiles": [{"name": "a", "config": {}}, {"name": "a", "config": {}}]},
                {"profiles": [{"name": "a", "config": {"aggression": 2.0}}]},
                {"profiles": [{"name": "a__baseline", "config": {}}]},
                {"detectors": [{"name": "d", "version": ""}]}, {"extra_field": 1}):
        with pytest.raises(ValidationError):
            small_exp(**bad)
    path = exp.to_file(tmp_path / "exp.json")
    assert ExperimentConfig.from_file(path) == exp
    assert ExperimentConfig.from_file(DEMO).name == "exp_demo"
    assert exp.config_hash() == small_exp(output_dir=str(tmp_path)).config_hash() != small_exp(seeds=[1]).config_hash()
    twins = small_exp(baseline=True).resolved_profiles()
    assert [p.name for p in twins] == ["robot", "robot__baseline", "pair", "pair__baseline"]
    assert twins[3].label == "baseline" and not twins[3].config.collusion and twins[3].config.humanlike_level == 0
    assert checkpoints(100) == [10, 20, 50, 100] and checkpoints(500) == [10, 20, 50, 100, 200, 500]
    assert checkpoints(300) == [10, 20, 50, 100, 200, 300]


# ---------------------------------------------------------------- runner


def test_runner_writes_runs_and_resumes_from_cache(tmp_path: Path) -> None:
    exp = small_exp()
    counter = Counting()
    res = run(tmp_path, exp, run_fn=counter)
    rows = lines(tmp_path / "out" / "runs.jsonl")
    assert len(rows) == 8 == len(res.records) and all(r["status"] == "ok" for r in rows)
    assert counter.calls == 4  # one table per (profile, seed), shared by both detectors
    assert {(r["profile_id"], r["detector_name"], r["seed"]) for r in rows} == {
        (p, d, s) for p in ("robot", "pair") for d in ("lo", "hi") for s in (1, 2)}
    assert all(r["wtsd"] is not None and len(r["prefix"]) == 4 for r in rows)
    again = Counting()
    res2 = run(tmp_path, exp, "out2", run_fn=again, resume=True)
    assert again.calls == 0 and res2.cached == 8 and all(r.cached for r in res2.records)
    fresh = Counting()
    run(tmp_path, exp, "out3", run_fn=fresh)  # without --resume everything is played again
    assert fresh.calls == 4


def test_failed_run_is_isolated_and_not_cached(tmp_path: Path) -> None:
    exp = small_exp()
    counter = Counting(fail=lambda c, s: c.collusion and s.seed == 2)
    res = run(tmp_path, exp, run_fn=counter)
    failed = [r for r in res.records if not r.ok]
    assert len(failed) == 2 and all(r.profile_id == "pair" and r.seed == 2 for r in failed)
    assert all("injected failure" in (r.error or "") for r in failed) and res.failed == 2
    assert sum(r.ok for r in res.records) == 6
    cache = RunCache(tmp_path / "cache")
    assert all(r.cache_key not in cache.results for r in failed)
    assert all(r.cache_key in cache.failures for r in failed)
    again = Counting()
    res2 = run(tmp_path, exp, "o2", run_fn=again, resume=True)  # known failure skipped, not reused as a result
    assert again.calls == 0 and res2.skipped_failed == 2 and sum(not r.ok for r in res2.records) == 2
    retry = Counting()
    res3 = run(tmp_path, exp, "o3", run_fn=retry, resume=True, retry_failed=True)
    assert retry.calls == 1 and all(r.ok for r in res3.records)
    assert not RunCache(tmp_path / "cache").failures  # success clears the failure


def test_dry_run_and_detector_filter(tmp_path: Path) -> None:
    counter = Counting()
    res = run(tmp_path, small_exp(), run_fn=counter, dry_run=True)
    assert counter.calls == 0 and res.tables_planned == 4 and not (tmp_path / "out").exists()
    res = run(tmp_path, small_exp(), run_fn=counter, detectors=["lo"])
    assert {r.detector_name for r in res.records} == {"lo"} and len(res.records) == 4
    with pytest.raises(ValueError):
        run(tmp_path, small_exp(), detectors=["missing"])


# --------------------------------------------------------------- metrics


def rec(profile: str, label: str, seed: int, is_bot: bool, conf: float, detector: str = "d",
        prefix: list[list[Any]] | None = None, status: str = "ok") -> RunRecord:
    return RunRecord(profile_id=profile, label=label, twin_of=None, detector_id=detector, detector_name=detector,
                     detector_version="v", seed=seed, n_hands=100, config_hash="h", cache_key=f"{profile}{seed}",
                     config={}, status=status,  # type: ignore[arg-type]
                     score={"is_bot": is_bot, "confidence": conf, "reason": "", "features": {}},
                     prefix=prefix or [], wtsd=50.0)


def test_detection_rate_fpr_matrix_and_ci() -> None:
    records = [rec("a", "bot", 1, True, 0.9), rec("a", "bot", 2, True, 0.4), rec("b", "bot", 1, False, 0.8),
               rec("b", "bot", 2, True, 0.7), rec("z", "baseline", 1, True, 0.6), rec("z", "baseline", 2, False, 0.9),
               rec("b", "bot", 3, True, 0.9, status="failed")]
    assert detection_rate(records, 0.5) == pytest.approx(0.5)  # 2 of 4 ok bot runs (0.4 < threshold)
    assert detection_rate(records, 0.3) == pytest.approx(0.75)
    assert false_positive_rate(records, 0.5) == pytest.approx(0.5)
    m = detection_matrix(records, 0.5)
    assert m["a"]["d"] == 0.5 and m["b"]["d"] == 0.5 and m["z"]["d"] == 0.5
    s = mean_ci([0.5, 0.7])
    assert s.mean == pytest.approx(0.6) and s.std == pytest.approx(math.sqrt(0.02))
    assert s.hi - s.mean == pytest.approx(12.706 * math.sqrt(0.02) / math.sqrt(2))
    assert mean_ci([0.3]).lo == 0.3 and mean_ci([]).n == 0
    summary = compute_summary(records, 0.5)
    row = summary.detectors[0]
    assert row["detection_rate"] == pytest.approx(0.5) and row["n_seeds"] == 2 and row["runs_failed"] == 1


def test_time_to_detect_is_stable_not_first() -> None:
    t, f = True, False
    assert time_to_detect([[10, t, .9], [20, f, .9], [50, t, .9], [100, t, .9]]) == 50
    assert time_to_detect([[10, t, .9], [20, t, .4], [50, t, .9], [100, t, .9]]) == 50  # low confidence = miss
    assert time_to_detect([[10, t, .9], [20, t, .9], [50, t, .9], [100, f, .9]]) is None
    assert time_to_detect([[10, t, .9], [20, t, .9]]) == 10
    assert time_to_detect([[10, f, .9], [20, t, .55], [50, t, .55]], threshold=0.6) is None
    assert time_to_detect([]) is None


# ---------------------------------------------------------------- report


def test_report_manifest_and_csvs(tmp_path: Path) -> None:
    res = run(tmp_path, small_exp(baseline=True), git="unknown")
    out = tmp_path / "out"
    for name in ("report.md", "summary.csv", "matrix.csv", "runs_flat.csv", "manifest.json", "runs.jsonl"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    exp = small_exp(baseline=True)
    assert manifest["config_hash"] == exp.config_hash() and manifest["git_hash"] == "unknown"
    assert set(manifest["versions"]) == {"python", "numpy", "pydantic", "pokermath", "harness"}
    assert manifest["versions"]["harness"] == __version__
    assert manifest["seeds"] == [1, 2] and manifest["batch_size"] == BATCH_SIZE and manifest["equity_samples"] == 50
    assert manifest["detectors"][0]["version"] == "v1" and manifest["detectors"][0]["stub"] is True
    assert manifest["started_at"] and manifest["ended_at"]
    flat = list(csv.DictReader((out / "runs_flat.csv").open()))
    assert len(flat) == len(res.records) == 16 and "wtsd" in flat[0]
    md = (out / "report.md").read_text()
    assert "FakeOracle stub" in md and "Top-5 worst" in md and "WTSD" in md and ">= 20 seeds" in md
    matrix = list(csv.reader((out / "matrix.csv").open()))
    assert len(matrix) == 1 + 4 and len(matrix[0]) == 3


def test_manifest_real_git_hash_or_unknown(tmp_path: Path) -> None:
    run(tmp_path, small_exp(seeds=[1], profiles=[{"name": "robot", "config": {}}]), git=None)
    gh = json.loads((tmp_path / "out" / "manifest.json").read_text())["git_hash"]
    assert gh == "unknown" or (len(gh) == 40 and all(c in "0123456789abcdef" for c in gh))


# ----------------------------------------------------------- determinism


def _stable_runs(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "timing"} for r in lines(path)]


def test_same_config_same_outputs(tmp_path: Path) -> None:
    exp = small_exp(baseline=True)
    run(tmp_path, exp, "a", cache_dir=tmp_path / "c1")
    run(tmp_path, exp, "b", cache_dir=tmp_path / "c2", workers=2)  # parallel == serial
    assert _stable_runs(tmp_path / "a" / "runs.jsonl") == _stable_runs(tmp_path / "b" / "runs.jsonl")
    for name in ("summary.csv", "matrix.csv", "runs_flat.csv"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text(), name


# --------------------------------------------------------------- safety


def test_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.com", 80))
    res = run(tmp_path, small_exp(seeds=[3]))
    assert res.records and all(r.ok for r in res.records)


# ------------------------------------------------------ cache invalidation


def test_cache_key_covers_every_part(tmp_path: Path) -> None:
    base = CacheKeyParts("p", "d@v#1", 1, 100, "abc", 200, BATCH_SIZE, __version__, "deadbeef", 6, (10, 100))
    changes: dict[str, Any] = {
        "profile_id": "q", "detector_id": "d@v2#1", "seed": 2, "n_hands": 200, "config_hash": "abd",
        "equity_samples": 300, "batch_size": BATCH_SIZE * 2, "harness_version": "9.9.9",
        "git_hash": "cafebabe", "n_seats": 5, "checkpoints": (10, 50, 100)}
    assert set(changes) == {f.name for f in fields(CacheKeyParts)}
    keys = {cache_key(base)} | {cache_key(replace(base, **{k: v})) for k, v in changes.items()}
    assert len(keys) == 1 + len(changes)
    assert cache_key(base) == cache_key(replace(base))

    exp = small_exp(seeds=[1], profiles=[{"name": "robot", "config": {}}], detectors=[{"name": "lo", "version": "v1"}])
    run(tmp_path, exp, run_fn=Counting())
    for variant, git in ((exp, "other-commit"), (small_exp(**{**_dump(exp), "equity_samples": 60}), "deadbeef"),
                         (small_exp(**{**_dump(exp), "seeds": [2]}), "deadbeef"),
                         (small_exp(**{**_dump(exp), "profiles": [{"name": "robot",
                                                                    "config": {"aggression": 0.6}}]}), "deadbeef"),
                         (small_exp(**{**_dump(exp), "detectors": [{"name": "lo", "version": "v2"}]}), "deadbeef")):
        counter = Counting()
        run(tmp_path, variant, run_fn=counter, resume=True, git=git)
        assert counter.calls == 1, variant
    counter = Counting()
    run(tmp_path, exp, run_fn=counter, resume=True)
    assert counter.calls == 0  # unchanged config still hits


def _dump(exp: ExperimentConfig) -> dict[str, Any]:
    return exp.model_dump(mode="json")


def test_cli_dry_run(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli_main(["run", "--config", str(DEMO), "--dry-run", "--cache-dir", str(tmp_path / "c")]) == 0
    out = capsys.readouterr().out
    assert "12 profiles x 2 detectors x 5 seeds = 120 runs" in out and "tables to play: 30" in out
