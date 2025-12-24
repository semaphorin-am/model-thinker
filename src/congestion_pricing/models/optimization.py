"""Toll schedule optimization model.

This model finds optimal toll schedules to maximize welfare
subject to constraints on revenue, equity, and political feasibility.
"""

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario, TollSchedule
from congestion_pricing.policy.results import ModelResult, AggregateResults
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@ModelRegistry.register("optimization")
class OptimizationModel(BaseModel):
    """Toll schedule optimization model.

    Finds optimal tolls to maximize social welfare subject to
    constraints. This is a bilevel problem where the lower level
    is the user equilibrium.

    Config parameters:
        objective: Objective function ('welfare', 'revenue', 'emissions')
        min_revenue: Minimum revenue constraint
        max_toll: Maximum toll level
        equity_weight: Weight on equity in objective
        solver: Optimization solver ('scipy', 'grid_search')
    """

    name = "optimization"
    version = "1.0.0"
    description = "Toll schedule optimization"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = False
    supports_dynamics = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.objective = self.config.get("objective", "welfare")
        self.min_revenue = self.config.get("min_revenue", 0.0)
        self.max_toll = self.config.get("max_toll", 25.0)
        self.equity_weight = self.config.get("equity_weight", 0.0)
        self.solver = self.config.get("solver", "grid_search")

    def run(self, scenario: Scenario) -> ModelResult:
        """Find optimal toll schedule.

        Args:
            scenario: Base scenario to optimize

        Returns:
            ModelResult with optimal toll and outcomes
        """
        start_time = time.time()
        logger.info(f"Running optimization model for scenario: {scenario.name}")

        # Find optimal toll
        if self.solver == "grid_search":
            optimal_toll, results = self._grid_search(scenario)
        else:
            optimal_toll, results = self._scipy_optimize(scenario)

        runtime = time.time() - start_time

        # Get aggregates for optimal toll
        aggregates = results["aggregates"]
        aggregates.runtime_seconds = runtime

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            parameters={
                **self.config,
                "optimal_toll": optimal_toll,
                "search_results": results.get("search_history", []),
            },
        )

    def _grid_search(
        self,
        scenario: Scenario,
    ) -> tuple[float, dict]:
        """Find optimal toll using grid search.

        Args:
            scenario: Base scenario

        Returns:
            Tuple of (optimal_toll, results_dict)
        """
        from congestion_pricing.models.equilibrium import EquilibriumModel
        from dataclasses import replace

        equilibrium = EquilibriumModel({"max_iterations": 50})

        toll_values = np.linspace(0, self.max_toll, 11)
        best_toll = 0.0
        best_objective = float("-inf")
        best_result = None
        search_history = []

        for toll in toll_values:
            # Create scenario with this toll
            new_toll = TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": toll},
            )
            modified = replace(scenario, toll=new_toll)

            # Run equilibrium
            result = equilibrium.run(modified)

            # Compute objective
            obj_value = self._compute_objective(result.aggregates, toll)

            # Check constraints
            feasible = self._check_constraints(result.aggregates)

            search_history.append({
                "toll": toll,
                "objective": obj_value,
                "feasible": feasible,
                "revenue": result.aggregates.toll_revenue,
            })

            if feasible and obj_value > best_objective:
                best_objective = obj_value
                best_toll = toll
                best_result = result

        if best_result is None:
            best_result = equilibrium.run(scenario)

        return best_toll, {
            "aggregates": best_result.aggregates,
            "search_history": search_history,
        }

    def _scipy_optimize(
        self,
        scenario: Scenario,
    ) -> tuple[float, dict]:
        """Find optimal toll using scipy optimization.

        Args:
            scenario: Base scenario

        Returns:
            Tuple of (optimal_toll, results_dict)
        """
        from scipy.optimize import minimize_scalar
        from congestion_pricing.models.equilibrium import EquilibriumModel
        from dataclasses import replace

        equilibrium = EquilibriumModel({"max_iterations": 50})
        cache = {}

        def objective(toll: float) -> float:
            toll = float(toll)
            if toll in cache:
                return cache[toll]

            new_toll = TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": toll},
            )
            modified = replace(scenario, toll=new_toll)
            result = equilibrium.run(modified)

            obj_value = -self._compute_objective(result.aggregates, toll)
            cache[toll] = obj_value

            return obj_value

        result = minimize_scalar(
            objective,
            bounds=(0, self.max_toll),
            method="bounded",
        )

        optimal_toll = result.x

        # Get final result
        new_toll = TollSchedule(
            toll_type="cordon",
            periods=["all_day"],
            rates={"all_day": optimal_toll},
        )
        modified = replace(scenario, toll=new_toll)
        final_result = equilibrium.run(modified)

        return optimal_toll, {"aggregates": final_result.aggregates}

    def _compute_objective(
        self,
        aggregates: AggregateResults,
        toll: float,
    ) -> float:
        """Compute objective function value.

        Args:
            aggregates: Model results
            toll: Toll level

        Returns:
            Objective value (higher is better)
        """
        if self.objective == "welfare":
            # Simplified welfare: revenue - travel time cost - emissions cost
            travel_time_cost = aggregates.total_vht * 20  # $20/hour VOT
            emissions_cost = aggregates.emissions_co2_kg * 0.05  # $50/ton CO2
            welfare = aggregates.toll_revenue - travel_time_cost - emissions_cost

            # Add equity penalty
            if self.equity_weight > 0:
                equity_penalty = abs(aggregates.welfare_change_q1) * self.equity_weight
                welfare -= equity_penalty

            return welfare

        elif self.objective == "revenue":
            return aggregates.toll_revenue

        elif self.objective == "emissions":
            return -aggregates.emissions_co2_kg

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def _check_constraints(self, aggregates: AggregateResults) -> bool:
        """Check if solution satisfies constraints.

        Args:
            aggregates: Model results

        Returns:
            True if feasible
        """
        if aggregates.toll_revenue < self.min_revenue:
            return False

        return True

    def pareto_frontier(
        self,
        scenario: Scenario,
        objectives: list[str] = ["revenue", "travel_time"],
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Compute Pareto frontier for multiple objectives.

        Args:
            scenario: Base scenario
            objectives: List of objectives to optimize
            n_points: Number of points on frontier

        Returns:
            DataFrame with Pareto-optimal solutions
        """
        from congestion_pricing.models.equilibrium import EquilibriumModel
        from dataclasses import replace

        equilibrium = EquilibriumModel({"max_iterations": 50})

        toll_values = np.linspace(0, self.max_toll, n_points)
        results = []

        for toll in toll_values:
            new_toll = TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": toll},
            )
            modified = replace(scenario, toll=new_toll)
            result = equilibrium.run(modified)

            results.append({
                "toll": toll,
                "revenue": result.aggregates.toll_revenue,
                "travel_time": result.aggregates.mean_travel_time,
                "emissions": result.aggregates.emissions_co2_kg,
                "vkt": result.aggregates.total_vkt,
            })

        df = pd.DataFrame(results)

        # Filter to Pareto-optimal points
        # (simplified - just return all for now)
        return df
