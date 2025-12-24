"""Demand response model using elasticity-based approach.

This is the simplest model in the framework, providing quick
back-of-envelope estimates of demand changes under pricing.
"""

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import (
    ModelResult,
    AggregateResults,
    LinkResults,
    ODResults,
)
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@ModelRegistry.register("demand_response")
class DemandResponseModel(BaseModel):
    """Elasticity-based demand response model.

    This model uses price elasticities to estimate changes in
    trip volumes and mode shares in response to congestion pricing.

    Config parameters:
        price_elasticity: Own-price elasticity for auto trips (default -0.3)
        cross_elasticity: Cross-price elasticity to transit (default 0.1)
        vot: Value of time in $/hour (default 20.0)
        time_elasticity: Elasticity with respect to travel time (default -0.5)
    """

    name = "demand_response"
    version = "1.0.0"
    description = "Elasticity-based demand response model"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = False
    supports_dynamics = False

    def __init__(self, config: dict | None = None):
        super().__init__(config)

        # Default parameters from literature
        self.price_elasticity = self.config.get("price_elasticity", -0.3)
        self.cross_elasticity = self.config.get("cross_elasticity", 0.1)
        self.vot = self.config.get("vot", 20.0)
        self.time_elasticity = self.config.get("time_elasticity", -0.5)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run demand response model.

        Args:
            scenario: Scenario to evaluate

        Returns:
            ModelResult with demand changes
        """
        import time
        start_time = time.time()

        logger.info(f"Running demand response model for scenario: {scenario.name}")

        # Load scenario data
        scenario.load()

        # Get baseline demand
        if scenario.od_matrix is None:
            raise ValueError("Scenario must have OD matrix loaded")

        od_df = scenario.od_matrix.copy()

        # Calculate toll impact
        if scenario.toll is not None:
            od_df = self._apply_toll_response(od_df, scenario)

        # Compute aggregates
        aggregates = self._compute_aggregates(od_df, scenario)
        aggregates.runtime_seconds = time.time() - start_time

        # Create OD results
        od_results = ODResults(data=od_df)

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            od_results=od_results,
        )

    def _apply_toll_response(
        self,
        od_df: pd.DataFrame,
        scenario: Scenario,
    ) -> pd.DataFrame:
        """Apply elasticity-based response to tolls.

        Args:
            od_df: OD demand matrix
            scenario: Scenario with toll definition

        Returns:
            Modified OD matrix
        """
        result = od_df.copy()

        # Get toll rate (simplified - use average across periods)
        toll_rate = np.mean(list(scenario.toll.rates.values()))

        # Convert toll to generalized cost change (in minutes)
        vot_per_minute = self.vot / 60
        gc_change_pct = toll_rate / (vot_per_minute * 30)  # Assume 30 min avg trip

        # Apply elasticity
        # delta_Q / Q = epsilon * delta_P / P
        auto_reduction = 1 + self.price_elasticity * gc_change_pct

        # Bound reduction to reasonable range
        auto_reduction = np.clip(auto_reduction, 0.5, 1.0)

        # Reduce auto demand
        if "demand_auto" in result.columns:
            result["demand_auto"] = result["demand_auto"] * auto_reduction

        # Increase transit demand (cross-elasticity)
        transit_increase = 1 + self.cross_elasticity * gc_change_pct
        if "demand_transit" in result.columns:
            result["demand_transit"] = result["demand_transit"] * transit_increase

        logger.info(
            f"Applied toll response: auto reduction={1-auto_reduction:.1%}, "
            f"transit increase={transit_increase-1:.1%}"
        )

        return result

    def _compute_aggregates(
        self,
        od_df: pd.DataFrame,
        scenario: Scenario,
    ) -> AggregateResults:
        """Compute aggregate metrics.

        Args:
            od_df: OD demand matrix
            scenario: Scenario configuration

        Returns:
            Aggregate results
        """
        # Total trips by mode
        total_auto = od_df.get("demand_auto", pd.Series([0])).sum()
        total_transit = od_df.get("demand_transit", pd.Series([0])).sum()
        total_other = od_df.get("demand_other", pd.Series([0])).sum()
        total_trips = total_auto + total_transit + total_other

        # Mode shares
        mode_share_auto = total_auto / total_trips if total_trips > 0 else 0
        mode_share_transit = total_transit / total_trips if total_trips > 0 else 0
        mode_share_other = total_other / total_trips if total_trips > 0 else 0

        # Estimate revenue
        toll_revenue = 0.0
        if scenario.toll is not None:
            avg_toll = np.mean(list(scenario.toll.rates.values()))
            # Assume some fraction of auto trips pay toll
            toll_revenue = total_auto * avg_toll * 0.5  # 50% in toll zone

        # Simple travel time estimate (placeholder)
        mean_travel_time = 25.0  # minutes

        # Simple emissions estimate
        avg_trip_length = 10.0  # km
        emissions_co2_kg = total_auto * avg_trip_length * 0.15  # 150 g/km

        return AggregateResults(
            mean_travel_time=mean_travel_time,
            mode_share_auto=mode_share_auto,
            mode_share_transit=mode_share_transit,
            mode_share_other=mode_share_other,
            toll_revenue=toll_revenue,
            emissions_co2_kg=emissions_co2_kg,
            converged=True,
            iterations=1,
        )

    def sensitivity_analysis(
        self,
        scenario: Scenario,
        toll_range: list[float],
    ) -> pd.DataFrame:
        """Run sensitivity analysis over toll levels.

        Args:
            scenario: Base scenario
            toll_range: List of toll values to test

        Returns:
            DataFrame with results for each toll level
        """
        from dataclasses import replace

        results = []

        for toll_level in toll_range:
            # Create modified scenario
            if scenario.toll is not None:
                new_rates = {k: toll_level for k in scenario.toll.rates}
                new_toll = replace(scenario.toll, rates=new_rates)
                modified = replace(scenario, toll=new_toll)
            else:
                from congestion_pricing.policy.scenario import TollSchedule
                new_toll = TollSchedule(
                    toll_type="cordon",
                    periods=["all_day"],
                    rates={"all_day": toll_level},
                )
                modified = replace(scenario, toll=new_toll)

            # Run model
            result = self.run(modified)

            results.append({
                "toll_level": toll_level,
                "mode_share_auto": result.aggregates.mode_share_auto,
                "mode_share_transit": result.aggregates.mode_share_transit,
                "toll_revenue": result.aggregates.toll_revenue,
                "emissions_co2_kg": result.aggregates.emissions_co2_kg,
            })

        return pd.DataFrame(results)
