"""Evaluation module for model validation and comparison.

This module provides metrics, validation, and comparison utilities
for congestion pricing models.
"""

from congestion_pricing.evaluation.metrics import (
    geh_statistic,
    rmse,
    mape,
    r_squared,
    compute_flow_metrics,
)
from congestion_pricing.evaluation.validation import (
    holdout_validation,
    cross_validate,
    temporal_split,
)
from congestion_pricing.evaluation.comparison import (
    compare_models,
    sensitivity_analysis,
    generate_comparison_report,
)

__all__ = [
    "geh_statistic",
    "rmse",
    "mape",
    "r_squared",
    "compute_flow_metrics",
    "holdout_validation",
    "cross_validate",
    "temporal_split",
    "compare_models",
    "sensitivity_analysis",
    "generate_comparison_report",
]
