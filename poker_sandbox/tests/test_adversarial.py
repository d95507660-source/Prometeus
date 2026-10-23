"""Tests for the M6 adversarial search (offline mock only; FakeOracle stands in for the real detector)."""

from __future__ import annotations

import csv
import random
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adapter.mock import MockAdapter
from adapter.models import HandResult
from adversarial import (
    AdversarialConfig,
    DetectorOracle,
    DetectorScore,
    Evaluation,
    Evaluator,
    FakeOracle,
    ParamSpace,
    RunOutput,
    RunSettings,
    bin_matrix,
    correlations,
    evolve,
    grid_configs,
    grid_search,
    random_configs,
    random_search,
    run_config,
    spearman,
    top_problematic,
    write_report,
)

SPACE = ParamSpace()
TINY = RunSettings(n_hands=25, equity_samples=50)


def fake_run(config: AdversarialConfig, settings: RunSettings) -> RunOutput:
    """No table at all: hands the config to the oracle (search mechanics only)."""
    return RunOutput([], {"target_seats": [0], "config": config}, [])


class ConfigOracle:
    """Smooth toy score of the config: low noise + low VPIP look like a bot."""

    def score(self, hand_history: list[HandResult], agent_meta: dict[str, Any],
              behavior_log: list[dict[str, Any]]) -> DetectorScore:
        c: AdversarialConfig = agent_meta["config"]
        p = min(1.0, max(0.0, 0.8 - 0.5 * c.humanlike_level * c.latency_sigma - 0.2 * (c.vpip_scale - 0.5)))
        return DetectorScore(p >= 0.5, p if p >= 0.5 else 1 - p, "toy", {"p": p})


class Counting:
    """Wraps a run function and counts calls."""

    def __init__(self, fn: Any = fake_run) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, config: AdversarialConfig, settings: RunSettings) -> RunOutput:
        self.calls += 1
        out: RunOutput = self.fn(config, settings)
        return out


def toy_evaluator(mode: Any = "detect", **kw: Any) -> Evaluator:
    return Evaluator(ConfigOracle(), TINY, mode=mode, run_fn=kw.pop("run_fn", fake_run), **kw)


# ----------------------------------------------------------------- space


def test_sample_and_mutate_stay_valid() -> None:
    rng = random.Random(0)
    configs = [SPACE.sample_uniform(rng) for _ in range(300)]
    for c in configs:
        SPACE.validate(c)
    assert {c.collusion for c in configs} == {False, True} and {c.group_size for c in configs} == {1, 2, 3}
    c = configs[0]
    for i in range(300):
        c = SPACE.mutate(c, rng, strength=0.3 if i % 2 else 0.05, jump_prob=0.2)
        SPACE.validate(c)
        SPACE.validate(SPACE.crossover(c, configs[i], rng))
    assert SPACE.mutate(configs[1], random.Random(1)) != configs[1]


def test_serialisation_roundtrip_and_validation() -> None:
    for c in [*random_configs(SPACE, 50, 3), SPACE.midpoint(), AdversarialConfig()]:
        assert AdversarialConfig.from_dict(c.to_dict()) == c
        assert AdversarialConfig.from_json(c.to_json()) == c
    assert AdversarialConfig.from_json('{"vpip_scale": 1}').vpip_scale == 1.0
    with pytest.raises(ValueError):
        AdversarialConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        AdversarialConfig(collusion=True, group_size=1)
    with pytest.raises(ValueError):
        AdversarialConfig(sizing="huge")
    with pytest.raises(ValueError):
        SPACE.validate(AdversarialConfig(aggression=1.5))
    # irrelevant knobs do not change the hash: coordination without collusion, noise at level 0
    base = AdversarialConfig()
    assert replace(base, coordination_strength=0.1, error_rate=0.1).config_hash() == base.config_hash()
    assert replace(base, humanlike_level=0.5).config_hash() != base.config_hash()


# ---------------------------------------------------------------- oracle


@pytest.fixture(scope="module")
def tiny_run() -> RunOutput:
    return run_config(SPACE.midpoint(collusion=True, group_size=2, humanlike_level=0.5), TINY)


def test_fake_oracle_is_stable(tiny_run: RunOutput) -> None:
    oracle = FakeOracle()
    assert isinstance(oracle, DetectorOracle)
    first = oracle.score(tiny_run.hand_history, tiny_run.agent_meta, tiny_run.behavior_log)
    second = oracle.score(tiny_run.hand_history, tiny_run.agent_meta, tiny_run.behavior_log)
    assert first == second and 0 <= first.confidence <= 1
    assert {"timing_cv", "vpip", "wtsd", "yield_rate", "bot_score"} <= set(first.features)
    flat = [dict(r, planned_delay_ms=1000.0) for r in tiny_run.behavior_log]
    robot = oracle.score(tiny_run.hand_history, tiny_run.agent_meta, flat)
    assert robot.features["s_timing"] == 1.0 and robot.bot_belief > first.bot_belief
    assert DetectorScore.from_dict(first.to_dict()).bot_belief == pytest.approx(first.bot_belief)
    with pytest.raises(ValueError):
        DetectorScore(True, 1.5)


def test_run_config_targets_and_meta(tiny_run: RunOutput) -> None:
    assert tiny_run.agent_meta["target_seats"] == [0, 1] and len(tiny_run.hand_history) == TINY.n_hands
    assert {r["seat"] for r in tiny_run.behavior_log} == {0, 1}


# ---------------------------------------------------------------- search


def test_grid_visits_every_point() -> None:
    extra = SPACE.midpoint(collusion=True, group_size=2, coordination_strength=0.8, signal_interval=2)
    points = grid_configs(SPACE, levels=(0.0, 0.33, 0.67, 1.0), extra=[extra])
    assert len(points) == 9 and len({p.config_hash() for p in points}) == 9
    assert all(p.coordination_strength == 1.0 and p.signal_interval == 1 for p in points[4:8])
    run = Counting()
    evals = grid_search(toy_evaluator(run_fn=run), points)
    assert [e.config for e in evals] == points and run.calls == 9 and all(e.phase == "grid" for e in evals)


def test_random_search_respects_budget() -> None:
    ev = toy_evaluator()
    evals = random_search(ev, SPACE, 17, seed=2)
    assert len(evals) == 17 == len(ev.evaluations) and ev.runs <= 17


def test_evolution_does_not_degrade() -> None:
    for mode in ("detect", "evade"):
        res = evolve(toy_evaluator(mode), SPACE, pop_size=6, generations=8, seed=4)
        best = res.best_per_generation()
        assert len(res.generations) == 8 and all(len(g) == 6 for g in res.generations)
        assert best == sorted(best) and best[-1] >= best[0]  # elitism: monotone
        assert res.best is not None and res.best.fitness(mode) == best[-1]


def test_cache_skips_reruns(tmp_path: Path) -> None:
    run = Counting()
    cache = tmp_path / "cache.jsonl"
    ev = toy_evaluator(run_fn=run, cache_path=cache, run_log_path=tmp_path / "runs.jsonl")
    c = SPACE.midpoint(humanlike_level=0.0)
    a, b = ev.evaluate(c), ev.evaluate(replace(c, error_rate=0.01))  # same effective config
    assert run.calls == 1 and not a.cached and b.cached and a.score == b.score
    ev.close()
    run2 = Counting()
    ev2 = toy_evaluator(run_fn=run2, cache_path=cache)
    assert ev2.evaluate(c).cached and run2.calls == 0
    other = Evaluator(ConfigOracle(), replace(TINY, seed=9), run_fn=run2, cache_path=cache)
    other.evaluate(c)  # other seed → other key
    assert run2.calls == 1
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2


# ---------------------------------------------------------------- report


def _prepared(n: int = 15) -> list[Evaluation]:
    out = []
    for i in range(n):
        c = AdversarialConfig(aggression=i / (n - 1), vpip_scale=1.0 + (i % 3) * 0.1)
        p = 0.1 + 0.8 * i / (n - 1)
        score = DetectorScore(p >= 0.5, p if p >= 0.5 else 1 - p, "prep",
                              {"s_wtsd": 1.0 if i % 2 else 0.0, "s_timing": 0.5, "wtsd": 50.0})
        out.append(Evaluation(c, score, TINY, 0.0, False, "prep"))
    return out


def test_report_correlations_top10_and_files(tmp_path: Path) -> None:
    evals = _prepared()
    corr = correlations(evals, SPACE, "detect")
    assert corr["aggression"] == pytest.approx(1.0) and correlations(evals, SPACE, "evade")["aggression"] < -0.99
    assert all(v != v for k, v in corr.items() if k == "bluff_freq")  # constant field → NaN
    top = top_problematic(evals, "detect")
    assert len(top) == 10 and top[0].config.aggression == 0.0
    assert [e.score.bot_belief for e in top] == sorted(e.score.bot_belief for e in top)
    assert top_problematic(evals, "evade")[0].config.aggression == 1.0
    matrix = bin_matrix(evals, SPACE)
    assert sum(c.n for c in matrix["aggression"]) == 15 and matrix["aggression"][2].caught == 1.0
    paths = write_report(evals, SPACE, "detect", tmp_path)
    rows = list(csv.DictReader(paths.csv.open()))
    assert len(rows) == 15 and "fitness_detect" in rows[0]
    md = paths.markdown.read_text()
    assert "Top-10" in md and "Spearman" in md and "WTSD" in md
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0) and spearman([1, 1, 1], [1, 2, 3]) != 0


# ----------------------------------------------------------- determinism


def test_determinism_of_configs_and_best() -> None:
    assert grid_configs(SPACE) == grid_configs(SPACE)
    assert random_configs(SPACE, 20, 5) == random_configs(SPACE, 20, 5) != random_configs(SPACE, 20, 6)
    best = [evolve(toy_evaluator(), SPACE, pop_size=5, generations=5, seed=1).best for _ in range(2)]
    assert best[0] is not None and best[1] is not None and best[0].config == best[1].config


def test_real_runs_are_deterministic() -> None:
    def search() -> tuple[list[DetectorScore], AdversarialConfig]:
        ev = Evaluator(FakeOracle(), TINY)
        grid = grid_search(ev, grid_configs(SPACE, levels=(0.0, 1.0)))
        res = evolve(ev, SPACE, pop_size=3, generations=2, seed=1)
        assert res.best is not None
        return [e.score for e in grid], res.best.config

    assert search() == search()


# --------------------------------------------------------------- safety


def test_no_network_and_mock_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    with pytest.raises(AssertionError, match="network"):  # the guard itself works
        socket.create_connection(("example.com", 80))

    tables = 0
    original = MockAdapter.connect

    def spy(self: MockAdapter, config: Any) -> None:
        nonlocal tables
        tables += 1
        original(self, config)

    monkeypatch.setattr(MockAdapter, "connect", spy)
    ev = Evaluator(FakeOracle(), TINY)
    grid_search(ev, grid_configs(SPACE, levels=(0.5,)))
    random_search(ev, SPACE, 2, seed=1)
    evolve(ev, SPACE, pop_size=2, generations=2, seed=1)
    assert ev.runs > 0 and tables == ev.runs  # every played scenario went through MockAdapter
