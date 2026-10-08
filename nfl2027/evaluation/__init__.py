"""Statistical evaluation metrics, out-of-fold cross-validation, and draft valuation."""

from nfl2027.evaluation.metrics import (
    evaluate_pipeline,
    run_benchmark_suite,
)
from nfl2027.evaluation.valuation import (
    expected_career_snaps,
    calculate_surplus_snaps,
    fit_draft_decay_curve,
    calculate_financial_surplus,
)

__all__ = [
    "evaluate_pipeline",
    "run_benchmark_suite",
    "expected_career_snaps",
    "calculate_surplus_snaps",
    "fit_draft_decay_curve",
    "calculate_financial_surplus",
]
