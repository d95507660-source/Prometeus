"""Search over :class:`ParamSpace` for configs the detector misses (or catches): grid, random, evolution.

Every evaluation plays the config on the mock with the same :class:`RunSettings` (one fixed seed for
all configs, so differences come from the config, not the deck) and asks the oracle for a score.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from adversarial.oracle import DetectorOracle, DetectorScore
from adversarial.runner import RunOutput, RunSettings, run_config
from adversarial.space import AdversarialConfig, ParamSpace

__all__ = [
    "Evaluation",
    "Evaluator",
    "EvolutionResult",
    "Mode",
    "RunFn",
    "evolve",
    "fitness",
    "grid_configs",
    "grid_search",
    "random_configs",
    "random_search",
]

Mode = Literal["evade", "detect"]
RunFn = Callable[[AdversarialConfig, RunSettings], RunOutput]


def fitness(score: DetectorScore, mode: Mode) -> float:
    """Every config is a bot. ``detect``: signed belief in "bot"; ``evade``: its negation (-1..1)."""
    if mode not in ("evade", "detect"):
        raise ValueError(f"unknown mode {mode!r}")
    return score.bot_belief if mode == "detect" else -score.bot_belief


@dataclass(slots=True)
class Evaluation:
    """One scored config."""

    config: AdversarialConfig
    score: DetectorScore
    settings: RunSettings
    elapsed_s: float
    cached: bool
    phase: str = "manual"
    generation: int | None = None
    adjustments: int = 0

    @property
    def config_hash(self) -> str:
        """Hash of the config's effective fields."""
        return self.config.config_hash()

    def fitness(self, mode: Mode) -> float:
        """See :func:`fitness`."""
        return fitness(self.score, mode)

    def to_log(self, mode: Mode) -> dict[str, Any]:
        """Run-log record (without ``ts``)."""
        return {"phase": self.phase, "generation": self.generation, "config_hash": self.config_hash,
                "config": self.config.to_dict(), **self.settings.as_dict(), "mode": mode,
                "score": self.score.to_dict(), "fitness": self.fitness(mode), "adjustments": self.adjustments,
                "elapsed_s": round(self.elapsed_s, 3), "cached": self.cached}


def _open_append(path: Path | None) -> IO[str] | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


class Evaluator:
    """Runs and scores configs with a cache keyed by (config effective hash, run settings)."""

    def __init__(self, oracle: DetectorOracle, settings: RunSettings | None = None, *, mode: Mode = "detect",
                 cache_path: Path | str | None = None, run_log_path: Path | str | None = None,
                 run_fn: RunFn = run_config, timer: Callable[[], float] = time.perf_counter) -> None:
        fitness(DetectorScore(True, 1.0), mode)  # validates mode early
        self.oracle = oracle
        self.settings = settings or RunSettings()
        self.mode: Mode = mode
        self.run_fn = run_fn
        self._timer = timer
        self._cache: dict[str, tuple[DetectorScore, float, int]] = {}
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = (DetectorScore.from_dict(row["score"]), row["elapsed_s"],
                                           row.get("adjustments", 0))
        self._cache_fh = _open_append(self.cache_path)
        self._log_fh = _open_append(Path(run_log_path) if run_log_path is not None else None)
        self.runs = 0  # scenario runs actually played (cache misses)
        self.evaluations: list[Evaluation] = []

    def cache_key(self, config: AdversarialConfig) -> str:
        """Key of ``config`` under the current settings."""
        raw = json.dumps({"config": config.config_hash(), **self.settings.as_dict()}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def evaluate(self, config: AdversarialConfig, *, phase: str = "manual",
                 generation: int | None = None) -> Evaluation:
        """Score ``config`` (from the cache when possible) and log the evaluation."""
        key = self.cache_key(config)
        hit = self._cache.get(key)
        if hit is not None:
            score, elapsed, adjustments = hit
            ev = Evaluation(config, score, self.settings, elapsed, True, phase, generation, adjustments)
        else:
            started = self._timer()
            out = self.run_fn(config, self.settings)
            score = self.oracle.score(out.hand_history, out.agent_meta, out.behavior_log)
            elapsed = self._timer() - started
            self.runs += 1
            self._cache[key] = (score, elapsed, out.adjustments)
            if self._cache_fh is not None:
                row = {"key": key, "config": config.to_dict(), **self.settings.as_dict(),
                       "score": score.to_dict(), "elapsed_s": elapsed, "adjustments": out.adjustments}
                self._cache_fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                self._cache_fh.flush()
            ev = Evaluation(config, score, self.settings, elapsed, False, phase, generation, out.adjustments)
        self.evaluations.append(ev)
        if self._log_fh is not None:
            record = {"ts": datetime.now(UTC).isoformat(), **ev.to_log(self.mode)}
            self._log_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._log_fh.flush()
        return ev

    def close(self) -> None:
        """Close the JSONL files. Idempotent."""
        for fh in (self._cache_fh, self._log_fh):
            if fh is not None:
                fh.close()
        self._cache_fh = self._log_fh = None


# ------------------------------------------------------------------ grid


def grid_configs(space: ParamSpace, *, collusion: Sequence[bool] = (False, True),
                 levels: Sequence[float] = (0.0, 0.5, 1.0),
                 extra: Iterable[AdversarialConfig] = ()) -> list[AdversarialConfig]:
    """Categorical axes x humanlike levels, continuous axes at their midpoints.

    Collusion points use the strong exploration knobs (strength 1, interval 1, a pair); add the
    M4 defaults through ``extra``.
    """
    out = []
    for on in collusion:
        for level in levels:
            coord: dict[str, Any] = {"coordination_strength": 1.0, "signal_interval": 1, "group_size": 2} if on \
                else {"group_size": 1}
            out.append(space.midpoint(collusion=on, humanlike_level=level, **coord))
    out.extend(extra)
    for c in out:
        space.validate(c)
    return out


def grid_search(evaluator: Evaluator, configs: Sequence[AdversarialConfig]) -> list[Evaluation]:
    """Evaluate every grid point in order."""
    return [evaluator.evaluate(c, phase="grid") for c in configs]


# ---------------------------------------------------------------- random


def random_configs(space: ParamSpace, n: int, seed: int) -> list[AdversarialConfig]:
    """``n`` uniform samples from a dedicated RNG."""
    rng = random.Random(seed)
    return [space.sample_uniform(rng) for _ in range(n)]


def random_search(evaluator: Evaluator, space: ParamSpace, n: int, seed: int) -> list[Evaluation]:
    """Exactly ``n`` evaluations (budget); duplicates are served by the cache."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return [evaluator.evaluate(c, phase="random") for c in random_configs(space, n, seed)]


# ------------------------------------------------------------- evolution


@dataclass(slots=True)
class EvolutionResult:
    """All generations (generation 0 = random init) and the best evaluation seen."""

    generations: list[list[Evaluation]] = field(default_factory=list)
    best: Evaluation | None = None
    mode: Mode = "detect"

    def best_per_generation(self) -> list[float]:
        """Best fitness of each generation."""
        return [max(e.fitness(self.mode) for e in gen) for gen in self.generations]


def evolve(evaluator: Evaluator, space: ParamSpace, *, pop_size: int = 5, generations: int = 10, seed: int = 1,
           mutation_strength: float = 0.2, jump_prob: float = 0.1, crossover_prob: float = 0.7, elite: int = 1,
           tournament: int = 2, initial: Sequence[AdversarialConfig] = ()) -> EvolutionResult:
    """Elitist GA: tournament selection, uniform crossover, Gaussian mutation; ``generations`` includes gen 0.

    Fitness follows ``evaluator.mode``; with elitism and a deterministic (cached) score the best
    fitness never decreases between generations.
    """
    if pop_size < 2 or generations < 1 or not 0 <= elite < pop_size or tournament < 1:
        raise ValueError("need pop_size >= 2, generations >= 1, 0 <= elite < pop_size, tournament >= 1")
    mode = evaluator.mode
    rng = random.Random(seed)
    population = list(initial)[:pop_size]
    population += [space.sample_uniform(rng) for _ in range(pop_size - len(population))]
    result = EvolutionResult(mode=mode)

    def rank(evals: list[Evaluation]) -> list[Evaluation]:
        return sorted(evals, key=lambda e: -e.fitness(mode))  # stable: ties keep population order

    for gen in range(generations):
        scored = [evaluator.evaluate(c, phase="evolution", generation=gen) for c in population]
        result.generations.append(scored)
        ranked = rank(scored)
        if result.best is None or ranked[0].fitness(mode) > result.best.fitness(mode):
            result.best = ranked[0]
        if gen == generations - 1:
            break

        def pick(pool: list[Evaluation] = scored) -> AdversarialConfig:
            contenders = [pool[rng.randrange(len(pool))] for _ in range(tournament)]
            return rank(contenders)[0].config

        children = [e.config for e in ranked[:elite]]
        while len(children) < pop_size:
            a, b = pick(), pick()
            child = space.crossover(a, b, rng) if rng.random() < crossover_prob else a
            children.append(space.mutate(child, rng, mutation_strength, jump_prob))
        population = children
    return result
