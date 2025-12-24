"""Tests for congestion pricing models."""

import pytest
import numpy as np

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario, TollSchedule


class TestModelRegistry:
    """Tests for model registry."""

    def test_register_and_get(self):
        """Test registering and retrieving a model."""
        # Import models to register them
        from congestion_pricing.models.demand_response import DemandResponseModel

        model_class = ModelRegistry.get("demand_response")
        assert model_class == DemandResponseModel

    def test_get_unknown_model(self):
        """Test getting unknown model raises error."""
        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")

    def test_list_models(self):
        """Test listing registered models."""
        # Import to register
        from congestion_pricing import models as _

        model_list = ModelRegistry.list_models()
        assert "demand_response" in model_list
        assert "equilibrium" in model_list


class TestDemandResponseModel:
    """Tests for demand response model."""

    def test_run_with_toll(self, sample_scenario):
        """Test running with toll."""
        from congestion_pricing.models.demand_response import DemandResponseModel

        model = DemandResponseModel()
        result = model.run(sample_scenario)

        assert result.aggregates.converged is True
        assert result.aggregates.mean_travel_time > 0

    def test_run_without_toll(self, no_toll_scenario):
        """Test running without toll."""
        from congestion_pricing.models.demand_response import DemandResponseModel

        model = DemandResponseModel()
        result = model.run(no_toll_scenario)

        assert result.aggregates.converged is True

    def test_toll_reduces_auto_share(self):
        """Test that tolls reduce auto mode share."""
        from congestion_pricing.models.demand_response import DemandResponseModel

        # No toll scenario
        no_toll = Scenario(name="no_toll", toll=None)

        # With toll
        toll = TollSchedule(
            toll_type="cordon",
            periods=["all_day"],
            rates={"all_day": 15.0},
        )
        with_toll = Scenario(name="with_toll", toll=toll)

        model = DemandResponseModel()
        result_no_toll = model.run(no_toll)
        result_with_toll = model.run(with_toll)

        # Auto share should be lower with toll
        assert result_with_toll.aggregates.mode_share_auto < result_no_toll.aggregates.mode_share_auto


class TestEquilibriumModel:
    """Tests for equilibrium model."""

    def test_run_basic(self, sample_scenario):
        """Test basic equilibrium run."""
        from congestion_pricing.models.equilibrium import EquilibriumModel

        model = EquilibriumModel({"max_iterations": 10})
        result = model.run(sample_scenario)

        assert result.model_name == "equilibrium"
        assert result.aggregates.mean_travel_time > 0

    def test_convergence(self, sample_scenario):
        """Test that model converges."""
        from congestion_pricing.models.equilibrium import EquilibriumModel

        model = EquilibriumModel({
            "max_iterations": 50,
            "convergence_threshold": 0.01,
        })
        result = model.run(sample_scenario)

        assert result.aggregates.converged is True


class TestOptimizationModel:
    """Tests for optimization model."""

    def test_run_basic(self, sample_scenario):
        """Test basic optimization run."""
        from congestion_pricing.models.optimization import OptimizationModel

        model = OptimizationModel({
            "generations": 5,
            "population_size": 10,
        })
        result = model.run(sample_scenario)

        assert result.model_name == "optimization"
        assert "optimal_toll" in result.parameters

    def test_optimal_toll_bounds(self, sample_scenario):
        """Test that optimal toll respects bounds."""
        from congestion_pricing.models.optimization import OptimizationModel

        model = OptimizationModel({
            "toll_min": 5.0,
            "toll_max": 20.0,
            "generations": 3,
        })
        result = model.run(sample_scenario)

        optimal = result.parameters.get("optimal_toll", 0)
        assert 5.0 <= optimal <= 20.0


class TestABMModel:
    """Tests for agent-based model."""

    def test_run_basic(self, sample_scenario):
        """Test basic ABM run."""
        from congestion_pricing.models.abm import ABMModel

        model = ABMModel({
            "n_agents": 100,
            "n_days": 10,
        })
        result = model.run(sample_scenario)

        assert result.model_name == "abm"
        assert result.aggregates.mean_travel_time > 0

    def test_has_trajectory(self, sample_scenario):
        """Test that ABM produces trajectory data."""
        from congestion_pricing.models.abm import ABMModel

        model = ABMModel({"n_agents": 50, "n_days": 5})
        result = model.run(sample_scenario)

        assert result.trajectory is not None
        assert len(result.trajectory) == 5


class TestStochasticModel:
    """Tests for stochastic model."""

    def test_run_basic(self, sample_scenario):
        """Test basic stochastic run."""
        from congestion_pricing.models.stochastic import StochasticModel

        model = StochasticModel({
            "n_replications": 5,
            "parallel": False,
        })
        result = model.run(sample_scenario)

        assert result.model_name == "stochastic"

    def test_uncertainty_quantification(self, sample_scenario):
        """Test that stochastic model produces uncertainty bounds."""
        from congestion_pricing.models.stochastic import StochasticModel

        model = StochasticModel({
            "n_replications": 10,
            "parallel": False,
        })
        result = model.run(sample_scenario)

        # Should have different percentiles
        assert result.aggregates.travel_time_p50 != result.aggregates.travel_time_p90


class TestDynamicModel:
    """Tests for dynamic/Markov model."""

    def test_run_basic(self, sample_scenario):
        """Test basic dynamic model run."""
        from congestion_pricing.models.dynamic import DynamicModel

        model = DynamicModel({"n_days": 10})
        result = model.run(sample_scenario)

        assert result.model_name == "dynamic"

    def test_trajectory_length(self, sample_scenario):
        """Test trajectory has correct length."""
        from congestion_pricing.models.dynamic import DynamicModel

        n_days = 20
        model = DynamicModel({"n_days": n_days})
        result = model.run(sample_scenario)

        assert len(result.trajectory) == n_days


class TestMLModel:
    """Tests for ML model."""

    def test_run_prediction(self, sample_scenario):
        """Test ML prediction task."""
        from congestion_pricing.models.ml import MLModel

        model = MLModel({"task": "predict"})
        result = model.run(sample_scenario)

        assert result.model_name == "ml"
        assert result.aggregates.mean_travel_time > 0

    def test_run_causal(self, sample_scenario):
        """Test ML causal inference task."""
        from congestion_pricing.models.ml import MLModel

        model = MLModel({"task": "causal"})
        result = model.run(sample_scenario)

        assert "treatment_effect" in result.parameters.get("model_metrics", {})


class TestSystemsModel:
    """Tests for systems synthesis model."""

    def test_run_basic(self, sample_scenario):
        """Test basic systems run."""
        from congestion_pricing.models.systems import SystemsModel

        model = SystemsModel({
            "models": ["demand_response"],
        })
        result = model.run(sample_scenario)

        assert result.model_name == "systems"

    def test_synthesis(self, sample_scenario):
        """Test that systems model synthesizes results."""
        from congestion_pricing.models.systems import SystemsModel

        model = SystemsModel({
            "models": ["demand_response", "equilibrium"],
            "synthesis_method": "mean",
        })
        result = model.run(sample_scenario)

        # Should have model comparison
        assert "model_comparison" in result.parameters
