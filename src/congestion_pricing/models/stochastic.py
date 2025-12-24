"""Stochastic model for reliability analysis.

This module provides Monte Carlo simulation for uncertainty
quantification and reliability metrics.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import ModelResult, AggregateResults
from congestion_pricing.utils.logging import get_logger
from congestion_pricing.utils.parallel import parallel_map

logger = get_logger(__name__)


@ModelRegistry.register("stochastic")
class StochasticModel(BaseModel):
    """Stochastic model for Monte Carlo reliability analysis.

    Runs multiple replications with random demand and capacity
    variations to quantify uncertainty in outcomes.

    Config parameters:
        n_replications: Number of Monte Carlo replications
        demand_cv: Coefficient of variation for demand
        incident_probability: Probability of incident per link
        parallel: Whether to run replications in parallel
    """

    name = "stochastic"
    version = "1.0.0"
    description = "Monte Carlo reliability analysis"

    supports_tolls = True
    supports_transit = False
    supports_stochastic = True
    supports_dynamics = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.n_replications = self.config.get("n_replications", 100)
        self.demand_cv = self.config.get("demand_cv", 0.1)
        self.incident_probability = self.config.get("incident_probability", 0.02)
        self.parallel = self.config.get("parallel", True)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run stochastic analysis.

        Args:
            scenario: Scenario to analyze

        Returns:
            ModelResult with uncertainty quantification
        """
        start_time = time.time()
        logger.info(
            f"Running stochastic model for scenario: {scenario.name} "
            f"({self.n_replications} replications)"
        )

        # Get n_replications from scenario if specified
        n_reps = scenario.n_replications or self.n_replications

        # Run replications
        if self.parallel:
            results = self._run_parallel(scenario, n_reps)
        else:
            results = self._run_sequential(scenario, n_reps)

        runtime = time.time() - start_time

        # Aggregate results
        aggregates, std_aggregates, percentiles = self._aggregate_results(results)
        aggregates.runtime_seconds = runtime
        aggregates.iterations = n_reps

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            aggregate_std=std_aggregates,
            aggregate_percentiles=percentiles,
        )

    def _run_single_replication(
        self,
        args: tuple[Scenario, int],
    ) -> AggregateResults:
        """Run a single Monte Carlo replication.

        Args:
            args: Tuple of (scenario, seed)

        Returns:
            Aggregate results for this replication
        """
        scenario, seed = args
        np.random.seed(seed)

        from congestion_pricing.models.equilibrium import EquilibriumModel
        from dataclasses import replace

        # Perturb demand
        demand_factor = np.random.lognormal(0, self.demand_cv)
        modified = replace(scenario, demand_multiplier=demand_factor)

        # Run equilibrium
        eq_model = EquilibriumModel({"max_iterations": 30})
        result = eq_model.run(modified)

        # Add random incident effect
        if np.random.random() < self.incident_probability * 10:  # Network-wide
            result.aggregates.mean_travel_time *= (1 + np.random.beta(2, 5))

        return result.aggregates

    def _run_sequential(
        self,
        scenario: Scenario,
        n_reps: int,
    ) -> list[AggregateResults]:
        """Run replications sequentially.

        Args:
            scenario: Scenario
            n_reps: Number of replications

        Returns:
            List of aggregate results
        """
        results = []
        for i in range(n_reps):
            result = self._run_single_replication((scenario, i))
            results.append(result)

            if (i + 1) % 20 == 0:
                logger.debug(f"Completed {i + 1}/{n_reps} replications")

        return results

    def _run_parallel(
        self,
        scenario: Scenario,
        n_reps: int,
    ) -> list[AggregateResults]:
        """Run replications in parallel.

        Args:
            scenario: Scenario
            n_reps: Number of replications

        Returns:
            List of aggregate results
        """
        args = [(scenario, i) for i in range(n_reps)]
        return parallel_map(self._run_single_replication, args, n_jobs=-1)

    def _aggregate_results(
        self,
        results: list[AggregateResults],
    ) -> tuple[AggregateResults, AggregateResults, dict[int, AggregateResults]]:
        """Aggregate results across replications.

        Args:
            results: List of replication results

        Returns:
            Tuple of (mean, std, percentiles)
        """
        # Extract metrics
        travel_times = [r.mean_travel_time for r in results]
        revenues = [r.toll_revenue for r in results]
        emissions = [r.emissions_co2_kg for r in results]
        vkts = [r.total_vkt for r in results]

        # Mean aggregates
        mean_agg = AggregateResults(
            mean_travel_time=np.mean(travel_times),
            travel_time_p50=np.median(travel_times),
            travel_time_p90=np.percentile(travel_times, 90),
            toll_revenue=np.mean(revenues),
            emissions_co2_kg=np.mean(emissions),
            total_vkt=np.mean(vkts),
            converged=True,
        )

        # Standard deviation
        std_agg = AggregateResults(
            mean_travel_time=np.std(travel_times),
            toll_revenue=np.std(revenues),
            emissions_co2_kg=np.std(emissions),
            total_vkt=np.std(vkts),
        )

        # Percentiles
        percentiles = {
            10: AggregateResults(
                mean_travel_time=np.percentile(travel_times, 10),
                toll_revenue=np.percentile(revenues, 10),
                emissions_co2_kg=np.percentile(emissions, 10),
            ),
            50: AggregateResults(
                mean_travel_time=np.percentile(travel_times, 50),
                toll_revenue=np.percentile(revenues, 50),
                emissions_co2_kg=np.percentile(emissions, 50),
            ),
            90: AggregateResults(
                mean_travel_time=np.percentile(travel_times, 90),
                toll_revenue=np.percentile(revenues, 90),
                emissions_co2_kg=np.percentile(emissions, 90),
            ),
        }

        return mean_agg, std_agg, percentiles

    def compute_reliability_metrics(
        self,
        results: list[AggregateResults],
    ) -> dict[str, float]:
        """Compute reliability metrics.

        Args:
            results: List of replication results

        Returns:
            Dictionary of reliability metrics
        """
        travel_times = [r.mean_travel_time for r in results]

        median_tt = np.median(travel_times)
        p95_tt = np.percentile(travel_times, 95)

        return {
            "buffer_time_index": (p95_tt - median_tt) / median_tt,
            "planning_time_index": p95_tt / np.percentile(travel_times, 50),
            "reliability_ratio": np.std(travel_times) / np.mean(travel_times),
        }
