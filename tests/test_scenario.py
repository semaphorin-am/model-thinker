"""Tests for scenario and results classes."""

import pytest
import tempfile
from pathlib import Path

from congestion_pricing.policy.scenario import (
    Scenario,
    TollSchedule,
    TransitScenario,
)
from congestion_pricing.policy.results import (
    ModelResult,
    AggregateResults,
)


class TestTollSchedule:
    """Tests for TollSchedule class."""

    def test_create_cordon_toll(self):
        """Test creating a cordon toll."""
        toll = TollSchedule(
            toll_type="cordon",
            periods=["peak", "off_peak"],
            rates={"peak": 15.0, "off_peak": 5.0},
        )

        assert toll.toll_type == "cordon"
        assert len(toll.periods) == 2
        assert toll.rates["peak"] == 15.0

    def test_get_rate_for_period(self):
        """Test getting toll rate for a period."""
        toll = TollSchedule(
            toll_type="cordon",
            periods=["peak", "off_peak"],
            rates={"peak": 15.0, "off_peak": 5.0},
        )

        assert toll.get_rate("peak") == 15.0
        assert toll.get_rate("off_peak") == 5.0
        assert toll.get_rate("unknown") == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        toll = TollSchedule(
            toll_type="cordon",
            periods=["peak"],
            rates={"peak": 10.0},
        )

        d = toll.to_dict()
        assert d["toll_type"] == "cordon"
        assert d["rates"]["peak"] == 10.0


class TestScenario:
    """Tests for Scenario class."""

    def test_create_scenario(self, sample_scenario):
        """Test creating a scenario."""
        assert sample_scenario.name == "test_scenario"
        assert sample_scenario.toll is not None
        assert sample_scenario.toll.toll_type == "cordon"

    def test_scenario_without_toll(self, no_toll_scenario):
        """Test scenario without toll."""
        assert no_toll_scenario.toll is None
        assert no_toll_scenario.demand_multiplier == 1.0

    def test_scenario_serialization(self, sample_scenario):
        """Test scenario to/from dict."""
        d = sample_scenario.to_dict()
        restored = Scenario.from_dict(d)

        assert restored.name == sample_scenario.name
        assert restored.toll.toll_type == sample_scenario.toll.toll_type

    def test_scenario_yaml_roundtrip(self, sample_scenario):
        """Test saving and loading scenario from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.yaml"
            sample_scenario.save(path)

            loaded = Scenario.load(path)
            assert loaded.name == sample_scenario.name


class TestAggregateResults:
    """Tests for AggregateResults class."""

    def test_create_results(self):
        """Test creating aggregate results."""
        results = AggregateResults(
            mean_travel_time=25.0,
            mode_share_auto=0.4,
            mode_share_transit=0.5,
            toll_revenue=100000.0,
            converged=True,
        )

        assert results.mean_travel_time == 25.0
        assert results.mode_share_auto == 0.4
        assert results.converged is True

    def test_results_defaults(self):
        """Test default values."""
        results = AggregateResults()

        assert results.mean_travel_time == 0.0
        assert results.converged is False


class TestModelResult:
    """Tests for ModelResult class."""

    def test_create_model_result(self, sample_scenario):
        """Test creating model result."""
        aggregates = AggregateResults(
            mean_travel_time=25.0,
            converged=True,
        )

        result = ModelResult(
            model_name="test_model",
            model_version="1.0.0",
            scenario_name=sample_scenario.name,
            aggregates=aggregates,
        )

        assert result.model_name == "test_model"
        assert result.aggregates.mean_travel_time == 25.0

    def test_result_save_load(self, sample_scenario):
        """Test saving and loading results."""
        aggregates = AggregateResults(mean_travel_time=25.0)
        result = ModelResult(
            model_name="test",
            model_version="1.0",
            scenario_name=sample_scenario.name,
            aggregates=aggregates,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(path)

            loaded = ModelResult.load(path)
            assert loaded.model_name == "test"
            assert loaded.aggregates.mean_travel_time == 25.0
