"""Model result definitions.

This module defines the standardized output structures for all models
in the framework, enabling consistent comparison and evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LinkResults:
    """Link-level model outputs.

    Attributes:
        data: DataFrame with columns edge_id, period, flow_vph, travel_time_min,
              speed_kmh, volume_capacity_ratio
    """

    data: pd.DataFrame

    def __post_init__(self):
        """Validate required columns exist."""
        required = ["edge_id", "period", "flow_vph", "travel_time_min"]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def edge_id(self) -> pd.Series:
        return self.data["edge_id"]

    @property
    def period(self) -> pd.Series:
        return self.data["period"]

    @property
    def flow_vph(self) -> pd.Series:
        return self.data["flow_vph"]

    @property
    def travel_time_min(self) -> pd.Series:
        return self.data["travel_time_min"]

    @property
    def speed_kmh(self) -> pd.Series:
        return self.data.get("speed_kmh", pd.Series(dtype=float))

    @property
    def volume_capacity_ratio(self) -> pd.Series:
        return self.data.get("volume_capacity_ratio", pd.Series(dtype=float))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.data.copy()

    def to_parquet(self, path: str | Path) -> None:
        """Save to Parquet file."""
        self.data.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "LinkResults":
        """Load from Parquet file."""
        return cls(data=pd.read_parquet(path))


@dataclass
class ODResults:
    """Origin-Destination level model outputs.

    Attributes:
        data: DataFrame with OD-level metrics
    """

    data: pd.DataFrame

    def __post_init__(self):
        """Validate required columns exist."""
        required = ["origin", "destination", "period"]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def origin(self) -> pd.Series:
        return self.data["origin"]

    @property
    def destination(self) -> pd.Series:
        return self.data["destination"]

    @property
    def period(self) -> pd.Series:
        return self.data["period"]

    @property
    def travel_time_auto(self) -> pd.Series:
        return self.data.get("travel_time_auto", pd.Series(dtype=float))

    @property
    def travel_time_transit(self) -> pd.Series:
        return self.data.get("travel_time_transit", pd.Series(dtype=float))

    @property
    def demand_auto(self) -> pd.Series:
        return self.data.get("demand_auto", pd.Series(dtype=float))

    @property
    def demand_transit(self) -> pd.Series:
        return self.data.get("demand_transit", pd.Series(dtype=float))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.data.copy()

    def to_parquet(self, path: str | Path) -> None:
        """Save to Parquet file."""
        self.data.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "ODResults":
        """Load from Parquet file."""
        return cls(data=pd.read_parquet(path))


@dataclass
class AggregateResults:
    """City/zone aggregate metrics.

    This class contains all the key performance indicators for a scenario,
    including efficiency, mode share, financial, environmental, and equity metrics.
    """

    # Efficiency metrics
    mean_travel_time: float = 0.0  # minutes, demand-weighted
    travel_time_std: float = 0.0  # standard deviation
    travel_time_p50: float = 0.0  # median
    travel_time_p90: float = 0.0  # 90th percentile
    total_vkt: float = 0.0  # vehicle-km traveled
    total_vht: float = 0.0  # vehicle-hours traveled
    throughput_cordon: float = 0.0  # trips/hour through charging zone
    avg_speed_kmh: float = 0.0  # network average speed

    # Mode share
    mode_share_auto: float = 0.0
    mode_share_transit: float = 0.0
    mode_share_other: float = 0.0

    # Financial metrics
    toll_revenue: float = 0.0  # daily revenue
    transit_fare_revenue: float = 0.0
    operating_cost: float = 0.0

    # Environmental metrics
    emissions_co2_kg: float = 0.0
    emissions_nox_kg: float = 0.0
    emissions_pm25_kg: float = 0.0

    # Equity metrics (by income quintile)
    welfare_change_q1: float = 0.0  # lowest income
    welfare_change_q2: float = 0.0
    welfare_change_q3: float = 0.0
    welfare_change_q4: float = 0.0
    welfare_change_q5: float = 0.0  # highest income
    accessibility_change_q1: float = 0.0  # jobs reachable change
    accessibility_change_q5: float = 0.0

    # Consumer surplus
    consumer_surplus_change: float = 0.0

    # Metadata
    converged: bool = True
    iterations: int = 0
    runtime_seconds: float = 0.0
    relative_gap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "efficiency": {
                "mean_travel_time": self.mean_travel_time,
                "travel_time_std": self.travel_time_std,
                "travel_time_p50": self.travel_time_p50,
                "travel_time_p90": self.travel_time_p90,
                "total_vkt": self.total_vkt,
                "total_vht": self.total_vht,
                "throughput_cordon": self.throughput_cordon,
                "avg_speed_kmh": self.avg_speed_kmh,
            },
            "mode_share": {
                "auto": self.mode_share_auto,
                "transit": self.mode_share_transit,
                "other": self.mode_share_other,
            },
            "financial": {
                "toll_revenue": self.toll_revenue,
                "transit_fare_revenue": self.transit_fare_revenue,
                "operating_cost": self.operating_cost,
            },
            "environmental": {
                "emissions_co2_kg": self.emissions_co2_kg,
                "emissions_nox_kg": self.emissions_nox_kg,
                "emissions_pm25_kg": self.emissions_pm25_kg,
            },
            "equity": {
                "welfare_change_q1": self.welfare_change_q1,
                "welfare_change_q2": self.welfare_change_q2,
                "welfare_change_q3": self.welfare_change_q3,
                "welfare_change_q4": self.welfare_change_q4,
                "welfare_change_q5": self.welfare_change_q5,
                "accessibility_change_q1": self.accessibility_change_q1,
                "accessibility_change_q5": self.accessibility_change_q5,
            },
            "consumer_surplus_change": self.consumer_surplus_change,
            "metadata": {
                "converged": self.converged,
                "iterations": self.iterations,
                "runtime_seconds": self.runtime_seconds,
                "relative_gap": self.relative_gap,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AggregateResults":
        """Deserialize from dictionary."""
        eff = data.get("efficiency", {})
        mode = data.get("mode_share", {})
        fin = data.get("financial", {})
        env = data.get("environmental", {})
        eq = data.get("equity", {})
        meta = data.get("metadata", {})

        return cls(
            mean_travel_time=eff.get("mean_travel_time", 0.0),
            travel_time_std=eff.get("travel_time_std", 0.0),
            travel_time_p50=eff.get("travel_time_p50", 0.0),
            travel_time_p90=eff.get("travel_time_p90", 0.0),
            total_vkt=eff.get("total_vkt", 0.0),
            total_vht=eff.get("total_vht", 0.0),
            throughput_cordon=eff.get("throughput_cordon", 0.0),
            avg_speed_kmh=eff.get("avg_speed_kmh", 0.0),
            mode_share_auto=mode.get("auto", 0.0),
            mode_share_transit=mode.get("transit", 0.0),
            mode_share_other=mode.get("other", 0.0),
            toll_revenue=fin.get("toll_revenue", 0.0),
            transit_fare_revenue=fin.get("transit_fare_revenue", 0.0),
            operating_cost=fin.get("operating_cost", 0.0),
            emissions_co2_kg=env.get("emissions_co2_kg", 0.0),
            emissions_nox_kg=env.get("emissions_nox_kg", 0.0),
            emissions_pm25_kg=env.get("emissions_pm25_kg", 0.0),
            welfare_change_q1=eq.get("welfare_change_q1", 0.0),
            welfare_change_q2=eq.get("welfare_change_q2", 0.0),
            welfare_change_q3=eq.get("welfare_change_q3", 0.0),
            welfare_change_q4=eq.get("welfare_change_q4", 0.0),
            welfare_change_q5=eq.get("welfare_change_q5", 0.0),
            accessibility_change_q1=eq.get("accessibility_change_q1", 0.0),
            accessibility_change_q5=eq.get("accessibility_change_q5", 0.0),
            consumer_surplus_change=data.get("consumer_surplus_change", 0.0),
            converged=meta.get("converged", True),
            iterations=meta.get("iterations", 0),
            runtime_seconds=meta.get("runtime_seconds", 0.0),
            relative_gap=meta.get("relative_gap", 0.0),
        )


@dataclass
class ModelResult:
    """Complete model output.

    This is the primary output object returned by all models in the framework.
    It contains the scenario that was run, model metadata, and all results
    at link, OD, and aggregate levels.

    Attributes:
        scenario_name: Name of the scenario that was run
        model_name: Name of the model that produced these results
        model_version: Version string of the model
        timestamp: When the model was run
        aggregates: Aggregate metrics
        link_results: Optional link-level results
        od_results: Optional OD-level results
        aggregate_std: Standard deviations for stochastic runs
        trajectory: Time series of daily aggregates for dynamic models
    """

    scenario_name: str
    model_name: str
    model_version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Results
    aggregates: AggregateResults = field(default_factory=AggregateResults)
    link_results: LinkResults | None = None
    od_results: ODResults | None = None

    # For stochastic models
    aggregate_std: AggregateResults | None = None
    aggregate_percentiles: dict[int, AggregateResults] = field(default_factory=dict)

    # For dynamic models
    trajectory: pd.DataFrame | None = None

    # Additional metadata
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> None:
        """Save all results to directory.

        Args:
            output_dir: Directory to save results to
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "scenario_name": self.scenario_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "warnings": self.warnings,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save aggregates
        with open(output_dir / "aggregates.json", "w") as f:
            json.dump(self.aggregates.to_dict(), f, indent=2)

        # Save link results
        if self.link_results is not None:
            self.link_results.to_parquet(output_dir / "link_results.parquet")

        # Save OD results
        if self.od_results is not None:
            self.od_results.to_parquet(output_dir / "od_results.parquet")

        # Save trajectory
        if self.trajectory is not None:
            self.trajectory.to_parquet(output_dir / "trajectory.parquet")

        # Save aggregate std for stochastic
        if self.aggregate_std is not None:
            with open(output_dir / "aggregates_std.json", "w") as f:
                json.dump(self.aggregate_std.to_dict(), f, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path) -> "ModelResult":
        """Load results from directory.

        Args:
            output_dir: Directory containing saved results

        Returns:
            Loaded ModelResult
        """
        import json

        output_dir = Path(output_dir)

        # Load metadata
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load aggregates
        with open(output_dir / "aggregates.json") as f:
            agg_data = json.load(f)
        aggregates = AggregateResults.from_dict(agg_data)

        # Load optional link results
        link_results = None
        link_file = output_dir / "link_results.parquet"
        if link_file.exists():
            link_results = LinkResults.from_parquet(link_file)

        # Load optional OD results
        od_results = None
        od_file = output_dir / "od_results.parquet"
        if od_file.exists():
            od_results = ODResults.from_parquet(od_file)

        # Load optional trajectory
        trajectory = None
        traj_file = output_dir / "trajectory.parquet"
        if traj_file.exists():
            trajectory = pd.read_parquet(traj_file)

        # Load optional aggregate std
        aggregate_std = None
        std_file = output_dir / "aggregates_std.json"
        if std_file.exists():
            with open(std_file) as f:
                aggregate_std = AggregateResults.from_dict(json.load(f))

        return cls(
            scenario_name=metadata["scenario_name"],
            model_name=metadata["model_name"],
            model_version=metadata.get("model_version", "1.0.0"),
            timestamp=metadata.get("timestamp", ""),
            aggregates=aggregates,
            link_results=link_results,
            od_results=od_results,
            aggregate_std=aggregate_std,
            trajectory=trajectory,
            parameters=metadata.get("parameters", {}),
            warnings=metadata.get("warnings", []),
        )

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = [
            f"Model Result: {self.model_name}",
            f"Scenario: {self.scenario_name}",
            f"Timestamp: {self.timestamp}",
            "",
            "Aggregate Metrics:",
            f"  Mean travel time: {self.aggregates.mean_travel_time:.1f} min",
            f"  Mode share (transit): {self.aggregates.mode_share_transit * 100:.1f}%",
            f"  Toll revenue: ${self.aggregates.toll_revenue:,.0f}/day",
            f"  CO2 emissions: {self.aggregates.emissions_co2_kg:,.0f} kg/day",
            "",
            "Convergence:",
            f"  Converged: {self.aggregates.converged}",
            f"  Iterations: {self.aggregates.iterations}",
            f"  Runtime: {self.aggregates.runtime_seconds:.1f}s",
        ]
        return "\n".join(lines)
