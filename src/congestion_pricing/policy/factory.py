"""Scenario factory for creating standard scenarios.

This module provides factory methods for creating common scenario configurations,
making it easy to set up baseline and policy scenarios for comparison.
"""

from pathlib import Path

from congestion_pricing.policy.scenario import (
    Scenario,
    ShockModel,
    TollSchedule,
    TransitScenario,
)


class ScenarioFactory:
    """Factory for creating standard scenarios."""

    @staticmethod
    def baseline(
        city: str,
        data_dir: str | Path = "data/processed",
    ) -> Scenario:
        """Create a no-toll baseline scenario.

        Args:
            city: City identifier (e.g., "london", "nyc")
            data_dir: Root directory for processed data

        Returns:
            Baseline scenario with no congestion pricing
        """
        data_dir = Path(data_dir)
        return Scenario(
            name=f"{city}_baseline",
            description=f"Baseline scenario for {city} with no congestion pricing",
            network_path=str(data_dir / city / "network"),
            od_matrix_path=str(data_dir / city / "demand" / "od_matrix.parquet"),
            city=city,
        )

    @staticmethod
    def simple_cordon(
        city: str,
        toll_rate: float,
        data_dir: str | Path = "data/processed",
    ) -> Scenario:
        """Create a simple cordon toll scenario.

        Args:
            city: City identifier
            toll_rate: Fixed toll rate in currency units
            data_dir: Root directory for processed data

        Returns:
            Scenario with fixed cordon toll
        """
        data_dir = Path(data_dir)
        return Scenario(
            name=f"{city}_cordon_{int(toll_rate)}",
            description=f"Cordon toll of ${toll_rate} for {city}",
            network_path=str(data_dir / city / "network"),
            od_matrix_path=str(data_dir / city / "demand" / "od_matrix.parquet"),
            city=city,
            toll=TollSchedule(
                toll_type="cordon",
                periods=["all_day"],
                rates={"all_day": toll_rate},
                zone_boundary=str(data_dir / city / "zones" / "cordon.geojson"),
            ),
        )

    @staticmethod
    def time_varying_cordon(
        city: str,
        peak_rate: float,
        shoulder_rate: float,
        off_peak_rate: float = 0.0,
        data_dir: str | Path = "data/processed",
    ) -> Scenario:
        """Create a time-varying cordon toll scenario.

        Args:
            city: City identifier
            peak_rate: Toll rate during peak periods
            shoulder_rate: Toll rate during shoulder periods
            off_peak_rate: Toll rate during off-peak (default 0)
            data_dir: Root directory for processed data

        Returns:
            Scenario with time-varying cordon toll
        """
        data_dir = Path(data_dir)
        return Scenario(
            name=f"{city}_timevar_{int(peak_rate)}_{int(shoulder_rate)}_{int(off_peak_rate)}",
            description=f"Time-varying toll for {city}: peak=${peak_rate}, "
            f"shoulder=${shoulder_rate}, off-peak=${off_peak_rate}",
            network_path=str(data_dir / city / "network"),
            od_matrix_path=str(data_dir / city / "demand" / "od_matrix.parquet"),
            city=city,
            toll=TollSchedule(
                toll_type="time_varying",
                periods=["am_peak", "shoulder_am", "midday", "shoulder_pm", "pm_peak", "evening"],
                rates={
                    "am_peak": peak_rate,
                    "shoulder_am": shoulder_rate,
                    "midday": off_peak_rate,
                    "shoulder_pm": shoulder_rate,
                    "pm_peak": peak_rate,
                    "evening": off_peak_rate,
                },
                zone_boundary=str(data_dir / city / "zones" / "cordon.geojson"),
            ),
            time_periods=["am_peak", "shoulder_am", "midday", "shoulder_pm", "pm_peak", "evening"],
        )

    @staticmethod
    def with_transit_investment(
        base_scenario: Scenario,
        frequency_multiplier: float = 1.5,
        investment_budget: float = 0.0,
    ) -> Scenario:
        """Add transit investment to an existing scenario.

        Args:
            base_scenario: Base scenario to modify
            frequency_multiplier: Factor to increase transit frequency
            investment_budget: Budget allocated for transit

        Returns:
            New scenario with transit improvements
        """
        from dataclasses import replace

        transit = TransitScenario(
            gtfs_path=str(Path(base_scenario.network_path).parent / "gtfs"),
            frequency_multiplier=frequency_multiplier,
        )

        return replace(
            base_scenario,
            name=f"{base_scenario.name}_transit_{int(frequency_multiplier * 100)}pct",
            description=f"{base_scenario.description} + transit investment",
            transit=transit,
            investment_budget=investment_budget,
        )

    @staticmethod
    def with_stochastic_shocks(
        base_scenario: Scenario,
        n_replications: int = 100,
        incident_probability: float = 0.02,
        demand_cv: float = 0.1,
    ) -> Scenario:
        """Add stochastic shocks to an existing scenario.

        Args:
            base_scenario: Base scenario to modify
            n_replications: Number of Monte Carlo replications
            incident_probability: Probability of incident per link
            demand_cv: Coefficient of variation for demand

        Returns:
            New scenario with stochastic elements
        """
        from dataclasses import replace

        shocks = ShockModel(
            incident_probability=incident_probability,
            demand_cv=demand_cv,
        )

        return replace(
            base_scenario,
            name=f"{base_scenario.name}_stochastic_{n_replications}",
            description=f"{base_scenario.description} + stochastic analysis",
            shocks=shocks,
            n_replications=n_replications,
        )

    @staticmethod
    def london_congestion_charge(
        toll_rate: float = 15.0,
        data_dir: str | Path = "data/processed",
    ) -> Scenario:
        """Create a London-style congestion charge scenario.

        Based on the actual London congestion charge parameters.

        Args:
            toll_rate: Daily charge (default 15 GBP as of 2024)
            data_dir: Root directory for processed data

        Returns:
            London congestion charge scenario
        """
        data_dir = Path(data_dir)
        return Scenario(
            name=f"london_cc_{int(toll_rate)}",
            description=f"London congestion charge at {toll_rate} GBP",
            network_path=str(data_dir / "london" / "network"),
            od_matrix_path=str(data_dir / "london" / "demand" / "od_matrix.parquet"),
            city="london",
            toll=TollSchedule(
                toll_type="time_varying",
                periods=["charging_hours", "free"],
                rates={
                    "charging_hours": toll_rate,  # 7am-6pm weekdays
                    "free": 0.0,
                },
                zone_boundary=str(data_dir / "london" / "zones" / "congestion_charge_zone.geojson"),
                exemptions=["electric", "disabled", "taxi", "bus"],
            ),
            time_periods=["charging_hours", "free"],
        )

    @staticmethod
    def nyc_cbd_toll(
        peak_rate: float = 15.0,
        off_peak_rate: float = 3.75,
        data_dir: str | Path = "data/processed",
    ) -> Scenario:
        """Create a NYC Central Business District toll scenario.

        Based on the MTA congestion pricing program parameters.

        Args:
            peak_rate: Peak period toll (weekday 5am-9pm, weekend 9am-9pm)
            off_peak_rate: Off-peak toll
            data_dir: Root directory for processed data

        Returns:
            NYC CBD toll scenario
        """
        data_dir = Path(data_dir)
        return Scenario(
            name=f"nyc_cbd_{int(peak_rate)}_{int(off_peak_rate)}",
            description=f"NYC CBD toll: peak=${peak_rate}, off-peak=${off_peak_rate}",
            network_path=str(data_dir / "nyc" / "network"),
            od_matrix_path=str(data_dir / "nyc" / "demand" / "od_matrix.parquet"),
            city="nyc",
            toll=TollSchedule(
                toll_type="time_varying",
                periods=["peak", "off_peak"],
                rates={
                    "peak": peak_rate,
                    "off_peak": off_peak_rate,
                },
                zone_boundary=str(data_dir / "nyc" / "zones" / "cbd.geojson"),
                exemptions=["emergency", "bus", "authorized_vehicles"],
            ),
            time_periods=["peak", "off_peak"],
        )
