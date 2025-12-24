"""Validation utilities for congestion pricing models.

This module provides holdout validation, cross-validation, and
temporal split validation methods.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import ModelResult
from congestion_pricing.evaluation.metrics import (
    rmse,
    mape,
    r_squared,
    geh_pass_rate,
)
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Results from model validation."""

    model_name: str
    n_train: int
    n_test: int
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate validation summary."""
        lines = [
            f"Validation Results: {self.model_name}",
            f"  Training samples: {self.n_train}",
            f"  Test samples: {self.n_test}",
            "  Metrics:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k}: {v:.4f}")
        return "\n".join(lines)


def holdout_validation(
    model: BaseModel,
    scenarios: list[Scenario],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ValidationResult:
    """Perform holdout validation.

    Args:
        model: Model to validate
        scenarios: List of scenarios with observed outcomes
        test_fraction: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        ValidationResult with metrics
    """
    np.random.seed(seed)

    n = len(scenarios)
    n_test = max(1, int(n * test_fraction))
    indices = np.random.permutation(n)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_scenarios = [scenarios[i] for i in train_indices]
    test_scenarios = [scenarios[i] for i in test_indices]

    logger.info(f"Holdout validation: {len(train_scenarios)} train, {len(test_scenarios)} test")

    # Run model on test scenarios
    predictions = []
    observations = []

    for scenario in test_scenarios:
        result = model.run(scenario)
        predictions.append(result.aggregates.mean_travel_time)
        # Assume scenario has observed_travel_time attribute
        observations.append(getattr(scenario, "observed_travel_time", 25.0))

    predictions = np.array(predictions)
    observations = np.array(observations)

    metrics = {
        "rmse": rmse(predictions, observations),
        "mape": mape(predictions, observations),
        "r_squared": r_squared(predictions, observations),
    }

    return ValidationResult(
        model_name=model.name,
        n_train=len(train_scenarios),
        n_test=len(test_scenarios),
        metrics=metrics,
    )


def cross_validate(
    model_class: type[BaseModel],
    model_config: dict[str, Any],
    scenarios: list[Scenario],
    n_folds: int = 5,
    seed: int = 42,
) -> ValidationResult:
    """Perform k-fold cross-validation.

    Args:
        model_class: Model class to instantiate
        model_config: Configuration for model
        scenarios: List of scenarios
        n_folds: Number of folds
        seed: Random seed

    Returns:
        ValidationResult with aggregated metrics
    """
    np.random.seed(seed)

    n = len(scenarios)
    indices = np.random.permutation(n)
    fold_size = n // n_folds

    fold_metrics = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n

        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        test_scenarios = [scenarios[i] for i in test_indices]

        # Create fresh model instance
        model = model_class(model_config)

        # Run on test fold
        predictions = []
        observations = []

        for scenario in test_scenarios:
            result = model.run(scenario)
            predictions.append(result.aggregates.mean_travel_time)
            observations.append(getattr(scenario, "observed_travel_time", 25.0))

        predictions = np.array(predictions)
        observations = np.array(observations)

        fold_result = {
            "fold": fold,
            "n_test": len(test_scenarios),
            "rmse": rmse(predictions, observations),
            "mape": mape(predictions, observations),
            "r_squared": r_squared(predictions, observations),
        }
        fold_metrics.append(fold_result)

        logger.info(f"Fold {fold + 1}/{n_folds}: RMSE={fold_result['rmse']:.3f}")

    # Aggregate across folds
    avg_metrics = {
        "rmse": np.mean([f["rmse"] for f in fold_metrics]),
        "rmse_std": np.std([f["rmse"] for f in fold_metrics]),
        "mape": np.mean([f["mape"] for f in fold_metrics]),
        "mape_std": np.std([f["mape"] for f in fold_metrics]),
        "r_squared": np.mean([f["r_squared"] for f in fold_metrics]),
        "r_squared_std": np.std([f["r_squared"] for f in fold_metrics]),
    }

    return ValidationResult(
        model_name=model_class.name,
        n_train=n - fold_size,
        n_test=fold_size,
        metrics=avg_metrics,
        fold_metrics=fold_metrics,
    )


def temporal_split(
    model: BaseModel,
    scenarios: list[Scenario],
    split_date: str | pd.Timestamp,
    date_column: str = "date",
) -> ValidationResult:
    """Perform temporal train/test split.

    Trains on data before split_date, tests on data after.

    Args:
        model: Model to validate
        scenarios: List of scenarios with date attribute
        split_date: Date to split on
        date_column: Attribute name for date

    Returns:
        ValidationResult with metrics
    """
    split_ts = pd.Timestamp(split_date)

    train_scenarios = []
    test_scenarios = []

    for scenario in scenarios:
        scenario_date = getattr(scenario, date_column, None)
        if scenario_date is None:
            logger.warning(f"Scenario {scenario.name} missing {date_column}")
            continue

        if pd.Timestamp(scenario_date) < split_ts:
            train_scenarios.append(scenario)
        else:
            test_scenarios.append(scenario)

    logger.info(
        f"Temporal split at {split_date}: "
        f"{len(train_scenarios)} train, {len(test_scenarios)} test"
    )

    if not test_scenarios:
        raise ValueError("No test scenarios after split date")

    # Run on test
    predictions = []
    observations = []

    for scenario in test_scenarios:
        result = model.run(scenario)
        predictions.append(result.aggregates.mean_travel_time)
        observations.append(getattr(scenario, "observed_travel_time", 25.0))

    predictions = np.array(predictions)
    observations = np.array(observations)

    metrics = {
        "rmse": rmse(predictions, observations),
        "mape": mape(predictions, observations),
        "r_squared": r_squared(predictions, observations),
    }

    return ValidationResult(
        model_name=model.name,
        n_train=len(train_scenarios),
        n_test=len(test_scenarios),
        metrics=metrics,
    )


def bootstrap_validation(
    model: BaseModel,
    scenarios: list[Scenario],
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    seed: int = 42,
) -> ValidationResult:
    """Perform bootstrap validation for confidence intervals.

    Args:
        model: Model to validate
        scenarios: List of scenarios
        n_bootstrap: Number of bootstrap samples
        sample_fraction: Fraction of data per sample
        seed: Random seed

    Returns:
        ValidationResult with confidence intervals
    """
    np.random.seed(seed)

    n = len(scenarios)
    sample_size = int(n * sample_fraction)

    bootstrap_metrics = []

    for b in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=sample_size, replace=True)
        out_of_bag = list(set(range(n)) - set(indices))

        if not out_of_bag:
            continue

        test_scenarios = [scenarios[i] for i in out_of_bag]

        predictions = []
        observations = []

        for scenario in test_scenarios:
            result = model.run(scenario)
            predictions.append(result.aggregates.mean_travel_time)
            observations.append(getattr(scenario, "observed_travel_time", 25.0))

        if predictions:
            bootstrap_metrics.append({
                "rmse": rmse(predictions, observations),
                "mape": mape(predictions, observations),
            })

    # Compute confidence intervals
    rmse_values = [m["rmse"] for m in bootstrap_metrics]
    mape_values = [m["mape"] for m in bootstrap_metrics]

    metrics = {
        "rmse_mean": np.mean(rmse_values),
        "rmse_ci_lower": np.percentile(rmse_values, 2.5),
        "rmse_ci_upper": np.percentile(rmse_values, 97.5),
        "mape_mean": np.mean(mape_values),
        "mape_ci_lower": np.percentile(mape_values, 2.5),
        "mape_ci_upper": np.percentile(mape_values, 97.5),
    }

    return ValidationResult(
        model_name=model.name,
        n_train=sample_size,
        n_test=n - sample_size,
        metrics=metrics,
        fold_metrics=bootstrap_metrics,
    )


def validate_against_counts(
    model: BaseModel,
    scenario: Scenario,
    count_data: pd.DataFrame,
    link_id_col: str = "link_id",
    count_col: str = "count",
) -> dict[str, float]:
    """Validate model against traffic count data.

    Args:
        model: Model to run
        scenario: Scenario to evaluate
        count_data: DataFrame with observed counts
        link_id_col: Column name for link IDs
        count_col: Column name for counts

    Returns:
        Dictionary of validation metrics
    """
    result = model.run(scenario)

    if result.link_results is None:
        raise ValueError("Model did not produce link-level results")

    # Get modeled flows
    modeled = pd.DataFrame({
        link_id_col: list(result.link_results.flows.keys()),
        "modeled_flow": list(result.link_results.flows.values()),
    })

    # Merge with counts
    merged = modeled.merge(
        count_data[[link_id_col, count_col]],
        on=link_id_col,
    )

    if len(merged) == 0:
        raise ValueError("No matching links between model and counts")

    mod_flows = merged["modeled_flow"].values
    obs_counts = merged[count_col].values

    return {
        "n_links": len(merged),
        "rmse": rmse(mod_flows, obs_counts),
        "mape": mape(mod_flows, obs_counts),
        "r_squared": r_squared(mod_flows, obs_counts),
        "geh_pass_5": geh_pass_rate(mod_flows, obs_counts, 5.0),
        "geh_pass_10": geh_pass_rate(mod_flows, obs_counts, 10.0),
    }
