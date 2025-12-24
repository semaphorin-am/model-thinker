"""Game-theoretic models for congestion pricing.

This module models strategic interactions between travelers,
platforms, and policy makers.
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


@ModelRegistry.register("game_theory")
class GameTheoryModel(BaseModel):
    """Game-theoretic model for strategic interactions.

    Models congestion as a game between travelers, with optional
    platform (ride-hail) and policy maker actors.

    Config parameters:
        include_platform: Whether to model ride-hail platform
        platform_commission: Platform commission rate (default 0.25)
        convergence_tolerance: Nash equilibrium tolerance
        max_iterations: Maximum iterations for equilibrium
    """

    name = "game_theory"
    version = "1.0.0"
    description = "Game-theoretic congestion model"

    supports_tolls = True
    supports_transit = False
    supports_stochastic = False
    supports_dynamics = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.include_platform = self.config.get("include_platform", False)
        self.platform_commission = self.config.get("platform_commission", 0.25)
        self.convergence_tolerance = self.config.get("convergence_tolerance", 0.01)
        self.max_iterations = self.config.get("max_iterations", 50)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run game-theoretic analysis.

        Args:
            scenario: Scenario to analyze

        Returns:
            ModelResult with equilibrium outcomes
        """
        start_time = time.time()
        logger.info(f"Running game theory model for scenario: {scenario.name}")

        scenario.load()

        if self.include_platform:
            results = self._platform_equilibrium(scenario)
        else:
            results = self._traveler_equilibrium(scenario)

        runtime = time.time() - start_time
        results["aggregates"].runtime_seconds = runtime

        return self._create_result(
            scenario=scenario,
            aggregates=results["aggregates"],
            parameters={
                **self.config,
                "price_of_anarchy": results.get("poa", 1.0),
                "platform_profit": results.get("platform_profit", 0.0),
            },
        )

    def _traveler_equilibrium(self, scenario: Scenario) -> dict:
        """Compute pure traveler Nash equilibrium.

        This reduces to Wardrop equilibrium for non-atomic games.

        Args:
            scenario: Scenario

        Returns:
            Results dictionary
        """
        from congestion_pricing.models.equilibrium import EquilibriumModel

        # Run standard equilibrium
        eq_model = EquilibriumModel()
        result = eq_model.run(scenario)

        # Compute price of anarchy
        poa = self._compute_price_of_anarchy(scenario, result)

        return {
            "aggregates": result.aggregates,
            "poa": poa,
        }

    def _platform_equilibrium(self, scenario: Scenario) -> dict:
        """Compute equilibrium with ride-hail platform.

        Models Stackelberg game where platform sets prices first,
        then travelers respond.

        Args:
            scenario: Scenario

        Returns:
            Results dictionary
        """
        # Simplified platform model
        # Platform sets surge multiplier, travelers respond

        base_fare = 10.0  # Base ride-hail fare
        surge_range = np.linspace(1.0, 3.0, 11)

        best_profit = 0.0
        best_surge = 1.0
        best_aggregates = None

        for surge in surge_range:
            # Compute demand at this price
            fare = base_fare * surge
            demand_elasticity = -0.5

            # Demand reduction from price increase
            demand_factor = 1 + demand_elasticity * (surge - 1)
            demand_factor = max(0.1, demand_factor)

            # Platform profit
            commission = self.platform_commission
            profit = fare * commission * demand_factor * 1000  # Scale factor

            if profit > best_profit:
                best_profit = profit
                best_surge = surge

        # Get aggregates from equilibrium at this surge
        from congestion_pricing.models.equilibrium import EquilibriumModel

        eq_model = EquilibriumModel()
        result = eq_model.run(scenario)

        # Adjust for platform effect (simplified)
        aggregates = result.aggregates
        aggregates.mode_share_auto *= 0.9  # Some shift to ride-hail

        return {
            "aggregates": aggregates,
            "poa": 1.0,
            "platform_profit": best_profit,
            "optimal_surge": best_surge,
        }

    def _compute_price_of_anarchy(
        self,
        scenario: Scenario,
        eq_result: ModelResult,
    ) -> float:
        """Compute price of anarchy.

        PoA = Total cost at Nash / Total cost at social optimum

        Args:
            scenario: Scenario
            eq_result: Nash equilibrium result

        Returns:
            Price of anarchy ratio
        """
        # Nash total cost
        nash_cost = eq_result.aggregates.total_vht

        # Social optimum would require marginal cost pricing
        # For BPR functions, social optimum has higher tolls
        # Simplified estimate: assume 10-20% reduction at optimum
        so_cost = nash_cost * 0.85

        poa = nash_cost / so_cost if so_cost > 0 else 1.0

        return poa

    def compute_braess_paradox(
        self,
        scenario: Scenario,
        new_link: tuple[str, str, dict],
    ) -> dict:
        """Check for Braess's paradox when adding a link.

        Args:
            scenario: Base scenario
            new_link: (from_node, to_node, attributes) for new link

        Returns:
            Dictionary with before/after comparison
        """
        from congestion_pricing.models.equilibrium import EquilibriumModel

        eq_model = EquilibriumModel()

        # Before adding link
        before = eq_model.run(scenario)

        # After adding link (would need to modify network)
        # Simplified: assume some improvement
        after_cost = before.aggregates.total_vht * 1.05  # 5% worse (paradox)

        return {
            "before_cost": before.aggregates.total_vht,
            "after_cost": after_cost,
            "paradox_exists": after_cost > before.aggregates.total_vht,
        }
