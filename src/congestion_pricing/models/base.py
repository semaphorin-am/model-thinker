"""Base model class for the congestion pricing framework.

All model implementations inherit from BaseModel to ensure
consistent interfaces and behavior.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import ModelResult, AggregateResults
from congestion_pricing.utils.io import load_yaml
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models.

    All models in the framework must implement this interface
    to ensure consistent usage and comparison.
    """

    # Model metadata
    name: str = "base"
    version: str = "1.0.0"
    description: str = "Base model class"

    # Model capabilities
    supports_tolls: bool = True
    supports_transit: bool = False
    supports_stochastic: bool = False
    supports_dynamics: bool = False

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration. Override in subclasses."""
        pass

    @classmethod
    def from_config_file(cls, path: str | Path) -> "BaseModel":
        """Load model from configuration file.

        Args:
            path: Path to YAML config file

        Returns:
            Initialized model
        """
        config = load_yaml(path)
        return cls(config)

    @abstractmethod
    def run(self, scenario: Scenario) -> ModelResult:
        """Run the model on a scenario.

        This is the main entry point for model execution.

        Args:
            scenario: Scenario to evaluate

        Returns:
            ModelResult with outputs
        """
        pass

    def validate_scenario(self, scenario: Scenario) -> list[str]:
        """Validate that a scenario is compatible with this model.

        Args:
            scenario: Scenario to validate

        Returns:
            List of validation error messages
        """
        errors = scenario.validate()

        if scenario.toll and not self.supports_tolls:
            errors.append(f"{self.name} does not support tolls")

        if scenario.transit and not self.supports_transit:
            errors.append(f"{self.name} does not support transit scenarios")

        if scenario.shocks and not self.supports_stochastic:
            errors.append(f"{self.name} does not support stochastic scenarios")

        if scenario.simulation_days > 1 and not self.supports_dynamics:
            errors.append(f"{self.name} does not support dynamic simulation")

        return errors

    def _create_result(
        self,
        scenario: Scenario,
        aggregates: AggregateResults,
        **kwargs: Any,
    ) -> ModelResult:
        """Create a ModelResult object.

        Args:
            scenario: The scenario that was run
            aggregates: Aggregate results
            **kwargs: Additional result fields

        Returns:
            ModelResult object
        """
        return ModelResult(
            scenario_name=scenario.name,
            model_name=self.name,
            model_version=self.version,
            aggregates=aggregates,
            parameters=self.config,
            **kwargs,
        )

    def _timed_run(self, func: callable, *args, **kwargs) -> tuple[Any, float]:
        """Run a function and measure execution time.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, runtime_seconds)
        """
        start = time.time()
        result = func(*args, **kwargs)
        runtime = time.time() - start
        return result, runtime

    def get_parameters(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        return self.config.copy()

    def set_parameter(self, key: str, value: Any) -> None:
        """Set a model parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self.config[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"


class ModelRegistry:
    """Registry for model classes."""

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model class.

        Args:
            name: Model name

        Returns:
            Decorator function
        """
        def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get a model class by name.

        Args:
            name: Model name

        Returns:
            Model class
        """
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List registered model names.

        Returns:
            List of model names
        """
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseModel:
        """Create a model instance by name.

        Args:
            name: Model name
            config: Model configuration

        Returns:
            Model instance
        """
        model_cls = cls.get(name)
        return model_cls(config)
