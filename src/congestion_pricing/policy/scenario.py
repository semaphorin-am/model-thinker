"""Scenario definitions for congestion pricing models.

This module defines the core data structures for specifying policy scenarios,
including toll schedules, transit alternatives, and stochastic shocks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


@dataclass
class TollSchedule:
    """Toll specification for congestion pricing.

    Attributes:
        toll_type: Type of toll - "cordon", "time_varying", or "link_based"
        periods: List of time period identifiers
        rates: Mapping from period (and optionally link) to toll rate
        zone_boundary: Path to GeoJSON file defining toll zone boundary
        exemptions: List of vehicle types exempt from tolls
    """

    toll_type: str  # "cordon", "time_varying", "link_based"
    periods: list[str]
    rates: dict[str, float]  # period -> rate, or "period:link_id" -> rate
    zone_boundary: str | None = None
    exemptions: list[str] = field(default_factory=list)

    def get_rate(self, period: str, link_id: str | None = None) -> float:
        """Get toll rate for a specific period and optional link.

        Args:
            period: Time period identifier
            link_id: Optional link identifier for link-based tolls

        Returns:
            Toll rate in currency units
        """
        if self.toll_type == "link_based" and link_id is not None:
            key = f"{period}:{link_id}"
            if key in self.rates:
                return self.rates[key]
        return self.rates.get(period, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "toll_type": self.toll_type,
            "periods": self.periods,
            "rates": self.rates,
            "zone_boundary": self.zone_boundary,
            "exemptions": self.exemptions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TollSchedule":
        """Deserialize from dictionary."""
        return cls(
            toll_type=data["toll_type"],
            periods=data["periods"],
            rates=data["rates"],
            zone_boundary=data.get("zone_boundary"),
            exemptions=data.get("exemptions", []),
        )


@dataclass
class TransitScenario:
    """Transit service level specification.

    Attributes:
        gtfs_path: Path to GTFS feed
        frequency_multiplier: Scale factor for service frequency (1.0 = baseline)
        fare_multiplier: Scale factor for fares (1.0 = baseline)
        capacity_multiplier: Scale factor for vehicle capacity
        new_routes: Optional list of new route definitions
    """

    gtfs_path: str
    frequency_multiplier: float = 1.0
    fare_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    new_routes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gtfs_path": self.gtfs_path,
            "frequency_multiplier": self.frequency_multiplier,
            "fare_multiplier": self.fare_multiplier,
            "capacity_multiplier": self.capacity_multiplier,
            "new_routes": self.new_routes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitScenario":
        """Deserialize from dictionary."""
        return cls(
            gtfs_path=data["gtfs_path"],
            frequency_multiplier=data.get("frequency_multiplier", 1.0),
            fare_multiplier=data.get("fare_multiplier", 1.0),
            capacity_multiplier=data.get("capacity_multiplier", 1.0),
            new_routes=data.get("new_routes", []),
        )


@dataclass
class ShockModel:
    """Stochastic shock specification for reliability analysis.

    Attributes:
        incident_probability: Probability of incident per link per time period
        severity_alpha: Beta distribution alpha parameter for severity
        severity_beta: Beta distribution beta parameter for severity
        demand_cv: Coefficient of variation for demand uncertainty
        weather_effects: Optional weather impact parameters
    """

    incident_probability: float = 0.02
    severity_alpha: float = 2.0
    severity_beta: float = 5.0
    demand_cv: float = 0.1
    weather_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "incident_probability": self.incident_probability,
            "severity_alpha": self.severity_alpha,
            "severity_beta": self.severity_beta,
            "demand_cv": self.demand_cv,
            "weather_effects": self.weather_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShockModel":
        """Deserialize from dictionary."""
        return cls(
            incident_probability=data.get("incident_probability", 0.02),
            severity_alpha=data.get("severity_alpha", 2.0),
            severity_beta=data.get("severity_beta", 5.0),
            demand_cv=data.get("demand_cv", 0.1),
            weather_effects=data.get("weather_effects", {}),
        )


@dataclass
class Scenario:
    """Complete scenario specification for model runs.

    This is the primary input object for all models in the framework.
    It encapsulates the network, demand, policy, and behavioral parameters.

    Attributes:
        name: Unique scenario identifier
        description: Human-readable description
        network_path: Path to processed network files
        od_matrix_path: Path to OD demand matrix
        city: City identifier (e.g., "london", "nyc")
        year: Reference year for the scenario
        toll: Optional toll schedule
        transit: Optional transit scenario
        investment_budget: Budget for transit investment
        vot_mean: Mean value of time ($/hour)
        vot_std: Standard deviation of value of time
        shocks: Optional stochastic shock model
        n_replications: Number of Monte Carlo replications
        time_periods: List of time periods to model
        simulation_days: Number of days for dynamic simulations
    """

    # Identifiers
    name: str
    description: str = ""

    # Network
    network_path: str = ""
    od_matrix_path: str = ""

    # City and time
    city: str = "london"
    year: int = 2024

    # Policy
    toll: TollSchedule | None = None
    transit: TransitScenario | None = None
    investment_budget: float = 0.0

    # Behavioral parameters
    vot_mean: float = 20.0  # $/hour
    vot_std: float = 10.0

    # Stochastic
    shocks: ShockModel | None = None
    n_replications: int = 1

    # Simulation parameters
    time_periods: list[str] = field(default_factory=lambda: ["am_peak", "pm_peak"])
    simulation_days: int = 100

    # Demand scaling
    demand_multiplier: float = 1.0

    # Loaded data (populated by load())
    _network: nx.DiGraph | None = field(default=None, repr=False)
    _od_matrix: pd.DataFrame | None = field(default=None, repr=False)
    _zones: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def network(self) -> nx.DiGraph | None:
        """Get loaded network graph."""
        return self._network

    @property
    def od_matrix(self) -> pd.DataFrame | None:
        """Get loaded OD matrix."""
        return self._od_matrix

    @property
    def zones(self) -> pd.DataFrame | None:
        """Get loaded zone definitions."""
        return self._zones

    def load(self) -> "Scenario":
        """Load data files into memory.

        Returns:
            Self for method chaining
        """
        if self.network_path:
            network_file = Path(self.network_path) / "graph.gpickle"
            if network_file.exists():
                self._network = nx.read_gpickle(network_file)

        if self.od_matrix_path:
            od_file = Path(self.od_matrix_path)
            if od_file.exists():
                self._od_matrix = pd.read_parquet(od_file)

        zones_file = Path(self.network_path) / "zones.parquet"
        if zones_file.exists():
            self._zones = pd.read_parquet(zones_file)

        return self

    def validate(self) -> list[str]:
        """Validate scenario configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.name:
            errors.append("Scenario name is required")

        if self.toll and self.toll.toll_type not in ["cordon", "time_varying", "link_based"]:
            errors.append(f"Invalid toll type: {self.toll.toll_type}")

        if self.vot_mean <= 0:
            errors.append("Value of time must be positive")

        if self.demand_multiplier <= 0:
            errors.append("Demand multiplier must be positive")

        if self.n_replications < 1:
            errors.append("Number of replications must be at least 1")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "network_path": self.network_path,
            "od_matrix_path": self.od_matrix_path,
            "city": self.city,
            "year": self.year,
            "toll": self.toll.to_dict() if self.toll else None,
            "transit": self.transit.to_dict() if self.transit else None,
            "investment_budget": self.investment_budget,
            "vot_mean": self.vot_mean,
            "vot_std": self.vot_std,
            "shocks": self.shocks.to_dict() if self.shocks else None,
            "n_replications": self.n_replications,
            "time_periods": self.time_periods,
            "simulation_days": self.simulation_days,
            "demand_multiplier": self.demand_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Deserialize from dictionary."""
        toll = TollSchedule.from_dict(data["toll"]) if data.get("toll") else None
        transit = TransitScenario.from_dict(data["transit"]) if data.get("transit") else None
        shocks = ShockModel.from_dict(data["shocks"]) if data.get("shocks") else None

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            network_path=data.get("network_path", ""),
            od_matrix_path=data.get("od_matrix_path", ""),
            city=data.get("city", "london"),
            year=data.get("year", 2024),
            toll=toll,
            transit=transit,
            investment_budget=data.get("investment_budget", 0.0),
            vot_mean=data.get("vot_mean", 20.0),
            vot_std=data.get("vot_std", 10.0),
            shocks=shocks,
            n_replications=data.get("n_replications", 1),
            time_periods=data.get("time_periods", ["am_peak", "pm_peak"]),
            simulation_days=data.get("simulation_days", 100),
            demand_multiplier=data.get("demand_multiplier", 1.0),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        """Load scenario from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save scenario to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
