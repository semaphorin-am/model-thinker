"""Evaluation metrics for congestion pricing models.

This module provides standard metrics for validating traffic
assignment and mode choice model outputs.
"""

from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


def geh_statistic(modeled: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute GEH statistic for traffic flow validation.

    The GEH (Geoffrey E. Havers) statistic is a standard metric
    for comparing modeled and observed traffic flows.

    GEH = sqrt(2 * (M - O)^2 / (M + O))

    Guidelines:
        - GEH < 5: Good fit for ~85% of links
        - GEH < 10: Acceptable

    Args:
        modeled: Modeled flow values
        observed: Observed flow values

    Returns:
        Array of GEH values
    """
    modeled = np.asarray(modeled)
    observed = np.asarray(observed)

    # Avoid division by zero
    denominator = modeled + observed
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return np.sqrt(2 * (modeled - observed) ** 2 / denominator)


def geh_pass_rate(
    modeled: np.ndarray,
    observed: np.ndarray,
    threshold: float = 5.0,
) -> float:
    """Compute percentage of links passing GEH threshold.

    Args:
        modeled: Modeled flow values
        observed: Observed flow values
        threshold: GEH threshold (default 5.0)

    Returns:
        Fraction of links with GEH < threshold
    """
    geh = geh_statistic(modeled, observed)
    return np.mean(geh < threshold)


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        predicted: Predicted values
        actual: Actual/observed values

    Returns:
        RMSE value
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    return np.sqrt(np.mean((predicted - actual) ** 2))


def mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        predicted: Predicted values
        actual: Actual/observed values (must be non-zero)

    Returns:
        MAPE as a fraction (0-1+)
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    # Filter out zeros in actual to avoid division errors
    mask = actual != 0
    if not mask.any():
        return np.inf

    return np.mean(np.abs((predicted[mask] - actual[mask]) / actual[mask]))


def r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination).

    Args:
        predicted: Predicted values
        actual: Actual/observed values

    Returns:
        R-squared value (can be negative for poor fits)
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        predicted: Predicted values
        actual: Actual/observed values

    Returns:
        MAE value
    """
    return np.mean(np.abs(np.asarray(predicted) - np.asarray(actual)))


def compute_flow_metrics(
    modeled_flows: pd.DataFrame,
    observed_flows: pd.DataFrame,
    link_id_col: str = "link_id",
    flow_col: str = "flow",
) -> dict[str, float]:
    """Compute comprehensive flow validation metrics.

    Args:
        modeled_flows: DataFrame with modeled link flows
        observed_flows: DataFrame with observed link flows
        link_id_col: Column name for link IDs
        flow_col: Column name for flow values

    Returns:
        Dictionary of validation metrics
    """
    # Merge on link ID
    merged = modeled_flows.merge(
        observed_flows,
        on=link_id_col,
        suffixes=("_modeled", "_observed"),
    )

    if len(merged) == 0:
        logger.warning("No matching links between modeled and observed flows")
        return {}

    modeled = merged[f"{flow_col}_modeled"].values
    observed = merged[f"{flow_col}_observed"].values

    return {
        "n_links": len(merged),
        "rmse": rmse(modeled, observed),
        "mape": mape(modeled, observed),
        "r_squared": r_squared(modeled, observed),
        "geh_mean": np.mean(geh_statistic(modeled, observed)),
        "geh_pass_rate_5": geh_pass_rate(modeled, observed, 5.0),
        "geh_pass_rate_10": geh_pass_rate(modeled, observed, 10.0),
        "total_modeled": np.sum(modeled),
        "total_observed": np.sum(observed),
        "flow_ratio": np.sum(modeled) / np.sum(observed) if np.sum(observed) > 0 else np.inf,
    }


def compute_mode_share_metrics(
    predicted_shares: dict[str, float],
    observed_shares: dict[str, float],
) -> dict[str, float]:
    """Compute mode share validation metrics.

    Args:
        predicted_shares: Dict mapping mode to predicted share
        observed_shares: Dict mapping mode to observed share

    Returns:
        Dictionary of validation metrics
    """
    common_modes = set(predicted_shares.keys()) & set(observed_shares.keys())

    if not common_modes:
        return {}

    pred = np.array([predicted_shares[m] for m in common_modes])
    obs = np.array([observed_shares[m] for m in common_modes])

    return {
        "n_modes": len(common_modes),
        "rmse": rmse(pred, obs),
        "mae": mae(pred, obs),
        "max_error": np.max(np.abs(pred - obs)),
        "per_mode_error": {
            m: predicted_shares[m] - observed_shares.get(m, 0)
            for m in common_modes
        },
    }


def compute_travel_time_metrics(
    modeled_times: np.ndarray,
    observed_times: np.ndarray,
) -> dict[str, float]:
    """Compute travel time validation metrics.

    Args:
        modeled_times: Modeled travel times
        observed_times: Observed travel times

    Returns:
        Dictionary of validation metrics
    """
    modeled = np.asarray(modeled_times)
    observed = np.asarray(observed_times)

    return {
        "n_observations": len(modeled),
        "rmse": rmse(modeled, observed),
        "mape": mape(modeled, observed),
        "r_squared": r_squared(modeled, observed),
        "mean_modeled": np.mean(modeled),
        "mean_observed": np.mean(observed),
        "std_modeled": np.std(modeled),
        "std_observed": np.std(observed),
        "bias": np.mean(modeled - observed),
        "bias_percent": np.mean((modeled - observed) / observed) if np.all(observed != 0) else np.inf,
    }


def convergence_gap(
    current_flows: np.ndarray,
    previous_flows: np.ndarray,
) -> float:
    """Compute relative gap between iterations.

    Used to assess convergence in equilibrium models.

    Args:
        current_flows: Current iteration flows
        previous_flows: Previous iteration flows

    Returns:
        Relative gap (RMSE / mean)
    """
    current = np.asarray(current_flows)
    previous = np.asarray(previous_flows)

    diff_rmse = rmse(current, previous)
    mean_flow = np.mean(current + previous) / 2

    if mean_flow == 0:
        return 0.0

    return diff_rmse / mean_flow


def elasticity_error(
    predicted_elasticity: float,
    observed_elasticity: float,
) -> dict[str, float]:
    """Compute error in demand elasticity estimation.

    Args:
        predicted_elasticity: Model-predicted elasticity
        observed_elasticity: Empirical elasticity

    Returns:
        Dictionary with error metrics
    """
    absolute_error = abs(predicted_elasticity - observed_elasticity)
    relative_error = (
        absolute_error / abs(observed_elasticity)
        if observed_elasticity != 0
        else np.inf
    )

    return {
        "predicted": predicted_elasticity,
        "observed": observed_elasticity,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def welfare_metrics(
    consumer_surplus_change: float,
    toll_revenue: float,
    external_benefits: float = 0.0,
) -> dict[str, float]:
    """Compute welfare evaluation metrics.

    Args:
        consumer_surplus_change: Change in consumer surplus
        toll_revenue: Revenue from tolling
        external_benefits: External benefits (emissions, safety)

    Returns:
        Dictionary with welfare metrics
    """
    total_welfare = consumer_surplus_change + toll_revenue + external_benefits

    return {
        "consumer_surplus_change": consumer_surplus_change,
        "toll_revenue": toll_revenue,
        "external_benefits": external_benefits,
        "total_welfare_change": total_welfare,
        "revenue_recycling_needed": -consumer_surplus_change if consumer_surplus_change < 0 else 0,
    }
