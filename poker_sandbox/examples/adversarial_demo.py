"""Grid + random search + evolution against FakeOracle (a stub; plug the real detector via DetectorOracle).

Synthetic bots only, offline mock, one fixed seed for every config; deterministic for a fixed --seed
(timings and log timestamps aside).

    python examples/adversarial_demo.py --seed 1 --hands 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script without installing the package

from adversarial import (  # noqa: E402
    Evaluator,
    FakeOracle,
    ParamSpace,
    RunSettings,
    correlations,
    evolve,
    grid_configs,
    grid_search,
    markdown_table,
    matrix_rows,
    random_search,
    ranked,
    top_rows,
    unique,
    write_report,
    wtsd_driven,
)

LEVELS = (0.0, 0.3333, 0.6667, 1.0)


def main() -> None:
    """Run the three search phases in "detect" mode and print the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--random", type=int, default=30)
    parser.add_argument("--pop", type=int, default=5)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "adversarial")
    args = parser.parse_args()

    cache, run_log = args.out / "cache.jsonl", args.out / "runs.jsonl"
    for p in (cache, run_log):
        p.unlink(missing_ok=True)  # fresh run: every config is actually played
    space = ParamSpace()
    evaluator = Evaluator(FakeOracle(), RunSettings(n_hands=args.hands, seed=args.seed), mode="detect",
                          cache_path=cache, run_log_path=run_log)
    started = time.perf_counter()
    default_collusion = space.midpoint(collusion=True, group_size=2, coordination_strength=0.8, signal_interval=2)
    grid = grid_search(evaluator, grid_configs(space, levels=LEVELS, extra=[default_collusion]))
    t_grid = time.perf_counter() - started
    rnd = random_search(evaluator, space, args.random, seed=args.seed)
    t_rnd = time.perf_counter() - started - t_grid
    evo = evolve(evaluator, space, pop_size=args.pop, generations=args.generations, seed=args.seed)
    t_evo = time.perf_counter() - started - t_grid - t_rnd
    evaluator.close()
    evals = evaluator.evaluations
    paths = write_report(evals, space, "detect", args.out)

    uniq = unique(evals)
    print(f"seed={args.seed} hands/config={args.hands} mode=detect  total={time.perf_counter() - started:.1f}s "
          f"(grid {t_grid:.1f}s, random {t_rnd:.1f}s, evolution {t_evo:.1f}s)")
    print(f"evaluations={len(evals)} scenario runs={evaluator.runs} cache hits={len(evals) - evaluator.runs} "
          f"unique={len(uniq)} flagged={sum(e.score.is_bot for e in uniq)}")
    print(f"grid: {len(grid)} points (8 + M4-default collusion), flagged {sum(e.score.is_bot for e in grid)}; "
          f"random: {len(rnd)}, flagged {sum(e.score.is_bot for e in rnd)}")
    print("evolution best fitness per generation:", " ".join(f"{f:+.3f}" for f in evo.best_per_generation()))
    if evo.best is not None:
        print(f"evolution best: {evo.best.config_hash} {evo.best.score.reason}")
    print("\nTop-10 problematic (missed bots, lowest bot belief first):")
    print(markdown_table(*top_rows(evals, "detect")))
    print("\nSpearman(field, fitness_detect):")
    print(markdown_table(["field", "spearman"], [[k, v] for k, v in ranked(correlations(evals, space, "detect"))]))
    print("\nField x bin (mean detect fitness / caught share, n):")
    print(markdown_table(*matrix_rows(evals, space)))
    wtsd = wtsd_driven(evals)
    print(f"\nFlagged mainly through WTSD: {len(wtsd)} configs: {', '.join(e.config_hash[:8] for e in wtsd)}")
    print(f"\nCSV:      {paths.csv}\nMarkdown: {paths.markdown}\nruns:     {run_log}\ncache:    {cache}")


if __name__ == "__main__":
    main()
