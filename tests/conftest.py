"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from congestion_pricing.policy.scenario import Scenario, TollSchedule, TransitScenario


@pytest.fixture
def sample_scenario():
    """Create a sample scenario for testing."""
    toll = TollSchedule(
        toll_type="cordon",
        periods=["am_peak", "pm_peak", "off_peak"],
        rates={"am_peak": 15.0, "pm_peak": 12.0, "off_peak": 5.0},
    )

    transit = TransitScenario(
        headway_factor=1.0,
        fare_multiplier=1.0,
    )

    return Scenario(
        name="test_scenario",
        description="Test scenario for unit tests",
        toll=toll,
        transit=transit,
        demand_multiplier=1.0,
    )


@pytest.fixture
def no_toll_scenario():
    """Create a scenario without tolls."""
    return Scenario(
        name="no_toll",
        description="Baseline scenario without congestion pricing",
        toll=None,
        transit=None,
        demand_multiplier=1.0,
    )


@pytest.fixture
def sample_flows():
    """Sample flow data for testing metrics."""
    np.random.seed(42)
    n_links = 100

    # Create correlated modeled and observed flows
    observed = np.random.exponential(500, n_links)
    noise = np.random.normal(0, 50, n_links)
    modeled = observed * 0.95 + noise

    return modeled, observed


@pytest.fixture
def sample_travel_times():
    """Sample travel time data for testing."""
    np.random.seed(42)
    n = 50

    observed = np.random.normal(25, 5, n)
    predicted = observed + np.random.normal(0, 2, n)

    return predicted, observed
