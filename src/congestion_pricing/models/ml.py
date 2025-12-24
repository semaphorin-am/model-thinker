"""Machine Learning models for prediction and causal inference.

This module provides ML-based prediction and causal inference
methods for congestion pricing analysis.
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


@ModelRegistry.register("ml")
class MLModel(BaseModel):
    """Machine Learning model for prediction and causal inference.

    Provides predictive models for travel time and congestion,
    plus causal inference methods (DiD, synthetic control).

    Config parameters:
        model_type: Type of model ('gradient_boosting', 'random_forest', 'linear')
        task: Task type ('predict', 'causal')
        features: List of feature columns
        target: Target column
    """

    name = "ml"
    version = "1.0.0"
    description = "ML prediction and causal inference"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = False
    supports_dynamics = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.model_type = self.config.get("model_type", "gradient_boosting")
        self.task = self.config.get("task", "predict")
        self.features = self.config.get("features", ["hour", "day_of_week", "toll"])
        self.target = self.config.get("target", "travel_time")

        self._model = None
        self._is_fitted = False

    def run(self, scenario: Scenario) -> ModelResult:
        """Run ML analysis.

        Args:
            scenario: Scenario to analyze

        Returns:
            ModelResult with predictions or causal estimates
        """
        start_time = time.time()
        logger.info(f"Running ML model for scenario: {scenario.name}")

        scenario.load()

        if self.task == "predict":
            results = self._run_prediction(scenario)
        elif self.task == "causal":
            results = self._run_causal_inference(scenario)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        runtime = time.time() - start_time
        results["aggregates"].runtime_seconds = runtime

        return self._create_result(
            scenario=scenario,
            aggregates=results["aggregates"],
            parameters={
                **self.config,
                "model_metrics": results.get("metrics", {}),
                "feature_importance": results.get("feature_importance", {}),
            },
        )

    def _run_prediction(self, scenario: Scenario) -> dict:
        """Run prediction task.

        Args:
            scenario: Scenario

        Returns:
            Results dictionary
        """
        # Generate synthetic training data
        X_train, y_train = self._generate_synthetic_data(n_samples=1000)

        # Fit model
        self.fit(X_train, y_train)

        # Make predictions for scenario
        toll_rate = 0.0
        if scenario.toll is not None:
            toll_rate = np.mean(list(scenario.toll.rates.values()))

        # Predict for each hour
        predictions = []
        for hour in range(6, 22):
            features = {
                "hour": hour,
                "day_of_week": 2,  # Wednesday
                "toll": toll_rate,
            }
            pred = self.predict(pd.DataFrame([features]))[0]
            predictions.append({"hour": hour, "predicted_travel_time": pred})

        predictions_df = pd.DataFrame(predictions)

        # Aggregate
        mean_tt = predictions_df["predicted_travel_time"].mean()

        return {
            "aggregates": AggregateResults(
                mean_travel_time=mean_tt,
                converged=True,
            ),
            "predictions": predictions_df,
            "metrics": self._compute_metrics(X_train, y_train),
            "feature_importance": self._get_feature_importance(),
        }

    def _run_causal_inference(self, scenario: Scenario) -> dict:
        """Run causal inference (Difference-in-Differences).

        Args:
            scenario: Scenario

        Returns:
            Results dictionary
        """
        # Generate synthetic panel data
        data = self._generate_did_data()

        # Run DiD regression
        treatment_effect = self._difference_in_differences(data)

        return {
            "aggregates": AggregateResults(
                mean_travel_time=25.0,  # Base
                converged=True,
            ),
            "metrics": {
                "treatment_effect": treatment_effect,
                "std_error": abs(treatment_effect) * 0.2,  # Simplified
            },
        }

    def _generate_synthetic_data(
        self,
        n_samples: int = 1000,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data.

        Args:
            n_samples: Number of samples

        Returns:
            Tuple of (features, target)
        """
        np.random.seed(42)

        data = {
            "hour": np.random.randint(6, 22, n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
            "toll": np.random.uniform(0, 20, n_samples),
        }

        df = pd.DataFrame(data)

        # Generate target: travel time depends on features
        # Peak hours have higher travel time
        hour_effect = np.where(
            (df["hour"] >= 8) & (df["hour"] <= 9), 10,
            np.where((df["hour"] >= 17) & (df["hour"] <= 18), 8, 0)
        )

        # Toll reduces travel time (less congestion)
        toll_effect = -0.3 * df["toll"]

        # Base travel time
        base = 25

        # Noise
        noise = np.random.normal(0, 3, n_samples)

        y = base + hour_effect + toll_effect + noise

        return df, pd.Series(y)

    def _generate_did_data(self) -> pd.DataFrame:
        """Generate synthetic DiD panel data.

        Returns:
            Panel DataFrame with treatment and control groups
        """
        np.random.seed(42)

        records = []
        n_units = 20
        n_periods = 24

        for unit in range(n_units):
            treated = unit < n_units // 2
            for period in range(n_periods):
                post = period >= n_periods // 2

                # Base outcome
                outcome = 30 + np.random.normal(0, 2)

                # Unit fixed effect
                outcome += unit * 0.5

                # Time trend
                outcome -= period * 0.1

                # Treatment effect (only for treated units post-treatment)
                if treated and post:
                    outcome -= 5  # Toll reduces travel time

                records.append({
                    "unit": unit,
                    "period": period,
                    "treated": treated,
                    "post": post,
                    "outcome": outcome,
                })

        return pd.DataFrame(records)

    def _difference_in_differences(self, data: pd.DataFrame) -> float:
        """Run Difference-in-Differences estimation.

        Args:
            data: Panel data with treated, post, outcome columns

        Returns:
            Treatment effect estimate
        """
        try:
            import statsmodels.formula.api as smf

            # DiD regression
            model = smf.ols(
                "outcome ~ treated * post + C(unit) + C(period)",
                data=data,
            ).fit()

            # Treatment effect is coefficient on treated:post interaction
            effect = model.params.get("treated:post", model.params.get("treated[T.True]:post[T.True]", 0))

            return float(effect)

        except ImportError:
            # Simple DiD without regression
            treated_post = data[(data["treated"]) & (data["post"])]["outcome"].mean()
            treated_pre = data[(data["treated"]) & (~data["post"])]["outcome"].mean()
            control_post = data[(~data["treated"]) & (data["post"])]["outcome"].mean()
            control_pre = data[(~data["treated"]) & (~data["post"])]["outcome"].mean()

            return (treated_post - treated_pre) - (control_post - control_pre)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModel":
        """Fit the prediction model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self for chaining
        """
        try:
            if self.model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                self._model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            elif self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                self._model = RandomForestRegressor(n_estimators=100, max_depth=5)
            else:
                from sklearn.linear_model import Ridge
                self._model = Ridge(alpha=1.0)

            self._model.fit(X, y)
            self._is_fitted = True

        except ImportError:
            logger.warning("scikit-learn not available, using simple model")
            self._model = None
            self._is_fitted = True
            self._simple_coeffs = {
                "intercept": y.mean(),
                "toll": -0.3,
            }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self._model is not None:
            return self._model.predict(X)
        else:
            # Simple prediction
            return (
                self._simple_coeffs["intercept"]
                + self._simple_coeffs.get("toll", 0) * X.get("toll", 0)
            ).values

    def _compute_metrics(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute model performance metrics.

        Args:
            X: Features
            y: True values

        Returns:
            Dictionary of metrics
        """
        if not self._is_fitted:
            return {}

        predictions = self.predict(X)

        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance
        """
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return {}

        importance = self._model.feature_importances_
        return dict(zip(self.features, importance))
