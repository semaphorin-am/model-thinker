"""Model comparison and sensitivity analysis.

This module provides utilities for comparing multiple models
and conducting sensitivity analyses.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario, TollSchedule
from congestion_pricing.policy.results import ModelResult
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Results from model comparison."""

    scenarios: list[str]
    models: list[str]
    metrics: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    def to_latex(self) -> str:
        """Export comparison table as LaTeX."""
        return self.metrics.to_latex(index=False, float_format="%.3f")

    def best_model(self, metric: str = "rmse", minimize: bool = True) -> str:
        """Identify best performing model for a metric."""
        if metric not in self.metrics.columns:
            raise ValueError(f"Metric {metric} not found")

        if minimize:
            idx = self.metrics[metric].idxmin()
        else:
            idx = self.metrics[metric].idxmax()

        return self.metrics.loc[idx, "model"]


def compare_models(
    models: list[BaseModel] | list[str],
    scenario: Scenario,
    metrics_to_compare: list[str] | None = None,
) -> ComparisonResult:
    """Compare multiple models on a single scenario.

    Args:
        models: List of model instances or model names
        scenario: Scenario to run
        metrics_to_compare: List of metrics to extract

    Returns:
        ComparisonResult with comparison table
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            "mean_travel_time",
            "mode_share_auto",
            "mode_share_transit",
            "toll_revenue",
            "emissions_co2_kg",
            "runtime_seconds",
        ]

    records = []

    for model in models:
        if isinstance(model, str):
            model_class = ModelRegistry.get(model)
            model = model_class()

        logger.info(f"Running {model.name} on {scenario.name}")

        try:
            result = model.run(scenario)
            record = {"model": model.name, "converged": result.aggregates.converged}

            for metric in metrics_to_compare:
                value = getattr(result.aggregates, metric, None)
                record[metric] = value

            records.append(record)

        except Exception as e:
            logger.warning(f"Failed to run {model.name}: {e}")
            record = {"model": model.name, "converged": False}
            for metric in metrics_to_compare:
                record[metric] = np.nan
            records.append(record)

    metrics_df = pd.DataFrame(records)

    # Compute summary statistics
    summary = {}
    for metric in metrics_to_compare:
        if metric in metrics_df.columns:
            values = metrics_df[metric].dropna()
            if len(values) > 1:
                summary[metric] = {
                    "mean": values.mean(),
                    "std": values.std(),
                    "cv": values.std() / values.mean() if values.mean() != 0 else np.inf,
                    "range": values.max() - values.min(),
                }

    return ComparisonResult(
        scenarios=[scenario.name],
        models=[m.name if isinstance(m, BaseModel) else m for m in models],
        metrics=metrics_df,
        summary=summary,
    )


def sensitivity_analysis(
    model: BaseModel,
    base_scenario: Scenario,
    parameter_name: str,
    parameter_values: list[float],
    output_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Run sensitivity analysis for a single parameter.

    Args:
        model: Model to run
        base_scenario: Base scenario to modify
        parameter_name: Name of parameter to vary
        parameter_values: List of values to test
        output_metrics: Metrics to extract from results

    Returns:
        DataFrame with parameter values and corresponding outputs
    """
    from dataclasses import replace

    if output_metrics is None:
        output_metrics = [
            "mean_travel_time",
            "mode_share_auto",
            "toll_revenue",
        ]

    records = []

    for value in parameter_values:
        # Create modified scenario
        if parameter_name == "toll_rate":
            new_toll = TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": value},
            )
            modified = replace(base_scenario, toll=new_toll)
        elif parameter_name == "demand_multiplier":
            modified = replace(base_scenario, demand_multiplier=value)
        elif hasattr(base_scenario, parameter_name):
            modified = replace(base_scenario, **{parameter_name: value})
        else:
            logger.warning(f"Unknown parameter: {parameter_name}")
            continue

        try:
            result = model.run(modified)

            record = {"parameter": parameter_name, "value": value}
            for metric in output_metrics:
                record[metric] = getattr(result.aggregates, metric, None)

            records.append(record)

        except Exception as e:
            logger.warning(f"Failed for {parameter_name}={value}: {e}")

    return pd.DataFrame(records)


def multi_parameter_sensitivity(
    model: BaseModel,
    base_scenario: Scenario,
    parameters: dict[str, list[float]],
    output_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Run multi-parameter sensitivity analysis.

    Creates a full factorial design over all parameter combinations.

    Args:
        model: Model to run
        base_scenario: Base scenario
        parameters: Dict mapping parameter names to value lists
        output_metrics: Metrics to extract

    Returns:
        DataFrame with all parameter combinations and outputs
    """
    from dataclasses import replace
    from itertools import product

    if output_metrics is None:
        output_metrics = ["mean_travel_time", "toll_revenue"]

    # Generate all combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(product(*param_values))

    records = []

    for combo in combinations:
        param_dict = dict(zip(param_names, combo))

        # Build modified scenario
        modified = base_scenario
        for name, value in param_dict.items():
            if name == "toll_rate":
                new_toll = TollSchedule(
                    toll_type="cordon",
                    periods=["all_day"],
                    rates={"all_day": value},
                )
                modified = replace(modified, toll=new_toll)
            elif hasattr(modified, name):
                modified = replace(modified, **{name: value})

        try:
            result = model.run(modified)

            record = param_dict.copy()
            for metric in output_metrics:
                record[metric] = getattr(result.aggregates, metric, None)

            records.append(record)

        except Exception as e:
            logger.warning(f"Failed for {param_dict}: {e}")

    return pd.DataFrame(records)


def compute_elasticities(
    sensitivity_results: pd.DataFrame,
    parameter_col: str = "value",
    base_value: float | None = None,
) -> dict[str, float]:
    """Compute elasticities from sensitivity analysis.

    Arc elasticity = (% change in output) / (% change in input)

    Args:
        sensitivity_results: DataFrame from sensitivity analysis
        parameter_col: Column with parameter values
        base_value: Base parameter value (uses first row if None)

    Returns:
        Dict mapping output metrics to elasticities
    """
    if base_value is None:
        base_value = sensitivity_results[parameter_col].iloc[0]

    base_row = sensitivity_results[
        sensitivity_results[parameter_col] == base_value
    ].iloc[0]

    elasticities = {}

    for col in sensitivity_results.columns:
        if col in ["parameter", parameter_col, "value"]:
            continue

        base_output = base_row[col]
        if base_output == 0 or base_value == 0:
            continue

        # Use last row as comparison point
        final_row = sensitivity_results.iloc[-1]
        final_value = final_row[parameter_col]
        final_output = final_row[col]

        pct_change_input = (final_value - base_value) / base_value
        pct_change_output = (final_output - base_output) / base_output

        if pct_change_input != 0:
            elasticities[col] = pct_change_output / pct_change_input

    return elasticities


def generate_comparison_report(
    comparison: ComparisonResult,
    output_path: str | None = None,
) -> str:
    """Generate markdown comparison report.

    Args:
        comparison: ComparisonResult from compare_models
        output_path: Optional path to save report

    Returns:
        Markdown report string
    """
    lines = [
        "# Model Comparison Report",
        "",
        f"## Scenarios: {', '.join(comparison.scenarios)}",
        f"## Models: {', '.join(comparison.models)}",
        "",
        "### Results Table",
        "",
        comparison.metrics.to_markdown(index=False),
        "",
        "### Summary Statistics",
        "",
    ]

    for metric, stats in comparison.summary.items():
        lines.append(f"**{metric}**:")
        lines.append(f"  - Mean: {stats['mean']:.3f}")
        lines.append(f"  - Std: {stats['std']:.3f}")
        lines.append(f"  - CV: {stats['cv']:.3f}")
        lines.append(f"  - Range: {stats['range']:.3f}")
        lines.append("")

    # Identify discrepancies (CV > 20%)
    discrepancies = []
    for metric, stats in comparison.summary.items():
        if stats["cv"] > 0.2:
            discrepancies.append(metric)

    if discrepancies:
        lines.extend([
            "### Discrepancies",
            "",
            "The following metrics show significant variation (CV > 20%) across models:",
            "",
        ])
        for d in discrepancies:
            lines.append(f"- {d}")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


def model_ranking(
    models: list[BaseModel],
    scenarios: list[Scenario],
    ranking_metric: str = "mean_travel_time",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Rank models across multiple scenarios.

    Args:
        models: List of models to compare
        scenarios: List of scenarios
        ranking_metric: Metric to use for ranking
        aggregation: How to aggregate across scenarios

    Returns:
        DataFrame with model rankings
    """
    all_results = []

    for scenario in scenarios:
        for model in models:
            try:
                result = model.run(scenario)
                all_results.append({
                    "model": model.name,
                    "scenario": scenario.name,
                    ranking_metric: getattr(result.aggregates, ranking_metric, np.nan),
                })
            except Exception as e:
                logger.warning(f"Failed {model.name} on {scenario.name}: {e}")

    df = pd.DataFrame(all_results)

    # Aggregate by model
    if aggregation == "mean":
        agg = df.groupby("model")[ranking_metric].mean()
    elif aggregation == "median":
        agg = df.groupby("model")[ranking_metric].median()
    else:
        agg = df.groupby("model")[ranking_metric].mean()

    ranking = agg.sort_values().reset_index()
    ranking["rank"] = range(1, len(ranking) + 1)

    return ranking
