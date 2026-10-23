"""Adversarial search (M6): fuzz bot configurations against a pluggable detector (DetectorOracle)."""

from adversarial.oracle import DetectorOracle, DetectorScore, FakeOracle
from adversarial.report import (
    BinCell,
    ReportPaths,
    bin_matrix,
    correlations,
    markdown_table,
    matrix_rows,
    ranked,
    spearman,
    top_problematic,
    top_rows,
    unique,
    write_report,
    wtsd_driven,
)
from adversarial.runner import RunOutput, RunSettings, run_config
from adversarial.search import (
    Evaluation,
    Evaluator,
    EvolutionResult,
    Mode,
    RunFn,
    evolve,
    fitness,
    grid_configs,
    grid_search,
    random_configs,
    random_search,
)
from adversarial.space import (
    DEFAULT_AXES,
    SIZING_SETS,
    AdversarialConfig,
    Axis,
    ChoiceAxis,
    FloatAxis,
    IntAxis,
    ParamSpace,
)

__all__ = [
    "DEFAULT_AXES", "SIZING_SETS", "AdversarialConfig", "Axis", "BinCell", "ChoiceAxis", "DetectorOracle",
    "DetectorScore", "Evaluation", "Evaluator", "EvolutionResult", "FakeOracle", "FloatAxis", "IntAxis", "Mode",
    "ParamSpace", "ReportPaths", "RunFn", "RunOutput", "RunSettings", "bin_matrix", "correlations", "evolve",
    "fitness", "grid_configs", "grid_search", "markdown_table", "matrix_rows", "random_configs", "random_search",
    "ranked", "run_config", "spearman", "top_problematic", "top_rows", "unique", "write_report", "wtsd_driven",
]
