"""Systems synthesis and cross-model integration.

This module provides orchestration for running multiple models
and synthesizing their results.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import ModelResult, AggregateResults
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@ModelRegistry.register("systems")
class SystemsModel(BaseModel):
    """Systems synthesis model for cross-model integration.

    Runs multiple model families and synthesizes their results
    to provide robust policy insights.

    Config parameters:
        models: List of models to run
        synthesis_method: How to combine results ('mean', 'median', 'ensemble')
        run_sensitivity: Whether to run sensitivity analysis
    """

    name = "systems"
    version = "1.0.0"
    description = "Cross-model synthesis and integration"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = True
    supports_dynamics = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.models = self.config.get(
            "models",
            ["demand_response", "equilibrium"]
        )
        self.synthesis_method = self.config.get("synthesis_method", "mean")
        self.run_sensitivity = self.config.get("run_sensitivity", False)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run systems synthesis.

        Args:
            scenario: Scenario to analyze

        Returns:
            ModelResult with synthesized outcomes
        """
        start_time = time.time()
        logger.info(f"Running systems synthesis for scenario: {scenario.name}")

        # Run each model
        model_results = self._run_models(scenario)

        # Synthesize results
        synthesized = self._synthesize_results(model_results)

        # Create comparison table
        comparison = self._create_comparison_table(model_results)

        # Identify discrepancies
        discrepancies = self._identify_discrepancies(comparison)

        runtime = time.time() - start_time
        synthesized.runtime_seconds = runtime

        return self._create_result(
            scenario=scenario,
            aggregates=synthesized,
            parameters={
                **self.config,
                "model_comparison": comparison.to_dict(),
                "discrepancies": discrepancies,
            },
        )

    def _run_models(self, scenario: Scenario) -> dict[str, ModelResult]:
        """Run all specified models.

        Args:
            scenario: Scenario to run

        Returns:
            Dictionary mapping model name to result
        """
        results = {}

        for model_name in self.models:
            try:
                model_class = ModelRegistry.get(model_name)
                model = model_class()
                result = model.run(scenario)
                results[model_name] = result
                logger.info(f"Completed {model_name}")
            except Exception as e:
                logger.warning(f"Failed to run {model_name}: {e}")

        return results

    def _synthesize_results(
        self,
        model_results: dict[str, ModelResult],
    ) -> AggregateResults:
        """Synthesize results from multiple models.

        Args:
            model_results: Dictionary of model results

        Returns:
            Synthesized aggregate results
        """
        if not model_results:
            return AggregateResults()

        # Extract metrics from each model
        metrics = {
            "mean_travel_time": [],
            "mode_share_auto": [],
            "mode_share_transit": [],
            "toll_revenue": [],
            "emissions_co2_kg": [],
        }

        for result in model_results.values():
            agg = result.aggregates
            metrics["mean_travel_time"].append(agg.mean_travel_time)
            metrics["mode_share_auto"].append(agg.mode_share_auto)
            metrics["mode_share_transit"].append(agg.mode_share_transit)
            metrics["toll_revenue"].append(agg.toll_revenue)
            metrics["emissions_co2_kg"].append(agg.emissions_co2_kg)

        # Synthesize based on method
        if self.synthesis_method == "mean":
            agg_func = np.mean
        elif self.synthesis_method == "median":
            agg_func = np.median
        else:
            agg_func = np.mean

        return AggregateResults(
            mean_travel_time=agg_func(metrics["mean_travel_time"]),
            mode_share_auto=agg_func(metrics["mode_share_auto"]),
            mode_share_transit=agg_func(metrics["mode_share_transit"]),
            toll_revenue=agg_func(metrics["toll_revenue"]),
            emissions_co2_kg=agg_func(metrics["emissions_co2_kg"]),
            converged=True,
        )

    def _create_comparison_table(
        self,
        model_results: dict[str, ModelResult],
    ) -> pd.DataFrame:
        """Create comparison table across models.

        Args:
            model_results: Dictionary of model results

        Returns:
            DataFrame comparing metrics across models
        """
        records = []

        for name, result in model_results.items():
            agg = result.aggregates
            records.append({
                "model": name,
                "mean_travel_time": agg.mean_travel_time,
                "mode_share_transit": agg.mode_share_transit,
                "toll_revenue": agg.toll_revenue,
                "emissions_co2_kg": agg.emissions_co2_kg,
                "converged": agg.converged,
                "runtime": agg.runtime_seconds,
            })

        return pd.DataFrame(records)

    def _identify_discrepancies(
        self,
        comparison: pd.DataFrame,
    ) -> list[dict]:
        """Identify significant discrepancies between models.

        Args:
            comparison: Comparison DataFrame

        Returns:
            List of discrepancy descriptions
        """
        discrepancies = []

        metrics = ["mean_travel_time", "mode_share_transit", "toll_revenue"]

        for metric in metrics:
            if metric not in comparison.columns:
                continue

            values = comparison[metric].values
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0

            if cv > 0.2:  # More than 20% coefficient of variation
                discrepancies.append({
                    "metric": metric,
                    "cv": cv,
                    "min": np.min(values),
                    "max": np.max(values),
                    "models_min": comparison.loc[comparison[metric].idxmin(), "model"],
                    "models_max": comparison.loc[comparison[metric].idxmax(), "model"],
                })

        return discrepancies

    def run_scenario_matrix(
        self,
        base_scenario: Scenario,
        toll_levels: list[float],
    ) -> pd.DataFrame:
        """Run all models across multiple toll levels.

        Args:
            base_scenario: Base scenario
            toll_levels: List of toll levels to test

        Returns:
            DataFrame with results for all model-toll combinations
        """
        from dataclasses import replace
        from congestion_pricing.policy.scenario import TollSchedule

        records = []

        for toll in toll_levels:
            # Create scenario with this toll
            new_toll = TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": toll},
            )
            scenario = replace(base_scenario, toll=new_toll)

            # Run all models
            model_results = self._run_models(scenario)

            for model_name, result in model_results.items():
                records.append({
                    "toll": toll,
                    "model": model_name,
                    "mean_travel_time": result.aggregates.mean_travel_time,
                    "mode_share_transit": result.aggregates.mode_share_transit,
                    "toll_revenue": result.aggregates.toll_revenue,
                    "emissions_co2_kg": result.aggregates.emissions_co2_kg,
                })

        return pd.DataFrame(records)

    def generate_systems_map(self) -> dict:
        """Generate a systems map of feedback loops.

        Returns:
            Dictionary describing system structure
        """
        return {
            "feedback_loops": [
                {
                    "name": "Congestion-Route Equilibrium",
                    "type": "balancing",
                    "variables": ["congestion", "route_choice", "link_flows"],
                },
                {
                    "name": "Induced Demand",
                    "type": "reinforcing",
                    "variables": ["travel_time", "trip_generation", "congestion"],
                },
                {
                    "name": "Transit Crowding",
                    "type": "balancing",
                    "variables": ["transit_ridership", "crowding", "mode_shift"],
                },
                {
                    "name": "Revenue-Investment",
                    "type": "reinforcing",
                    "variables": ["toll_revenue", "transit_investment", "transit_quality"],
                },
            ],
            "scenario_dimensions": {
                "induced_demand": {"low": 0, "medium": 0.2, "high": 0.5},
                "transit_investment": {"low": 0.25, "medium": 0.5, "high": 0.75},
                "political_constraint": {"low": 5, "medium": 15, "high": 25},
            },
        }
