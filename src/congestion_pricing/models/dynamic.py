"""Dynamic/Markov model for day-to-day adaptation.

This module models the day-to-day evolution of traveler choices
as a Markov process over choice states.
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


@ModelRegistry.register("dynamic")
class DynamicModel(BaseModel):
    """Markov model for day-to-day dynamics.

    Models aggregate behavior as transitions between discrete
    choice states (mode, departure time).

    Config parameters:
        n_days: Number of simulation days
        inertia: Probability of staying in current state
        logit_scale: Scale parameter for logit choice
    """

    name = "dynamic"
    version = "1.0.0"
    description = "Markov day-to-day adaptation model"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = False
    supports_dynamics = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.n_days = self.config.get("n_days", 100)
        self.inertia = self.config.get("inertia", 0.7)
        self.logit_scale = self.config.get("logit_scale", 0.1)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run dynamic adaptation model.

        Args:
            scenario: Scenario to simulate

        Returns:
            ModelResult with adaptation trajectory
        """
        start_time = time.time()
        logger.info(f"Running dynamic model for scenario: {scenario.name}")

        scenario.load()

        # Define states: (mode, departure_hour)
        modes = ["auto", "transit"]
        departures = [7, 8, 9]  # AM peak hours
        states = [(m, d) for m in modes for d in departures]

        # Initial distribution (uniform)
        n_states = len(states)
        distribution = np.ones(n_states) / n_states

        # Get costs for each state
        base_costs = self._compute_state_costs(states, scenario)

        # Simulate day-to-day dynamics
        trajectory = self._simulate_dynamics(states, distribution, base_costs, scenario)

        runtime = time.time() - start_time

        # Final distribution (average of last 10 days)
        final_dist = trajectory.iloc[-10:][
            [f"state_{i}" for i in range(n_states)]
        ].mean()

        # Compute aggregates from final distribution
        aggregates = self._distribution_to_aggregates(final_dist, states)
        aggregates.runtime_seconds = runtime

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            trajectory=trajectory,
        )

    def _compute_state_costs(
        self,
        states: list[tuple[str, int]],
        scenario: Scenario,
    ) -> np.ndarray:
        """Compute cost for each state.

        Args:
            states: List of (mode, departure) states
            scenario: Scenario with toll info

        Returns:
            Array of costs per state
        """
        costs = []

        toll_rate = 0.0
        if scenario.toll is not None:
            toll_rate = np.mean(list(scenario.toll.rates.values()))

        for mode, departure in states:
            # Base travel time
            if mode == "auto":
                # Peak hour congestion
                if departure == 8:
                    base_time = 30.0
                else:
                    base_time = 25.0

                # Add toll
                cost = base_time + toll_rate / (20 / 60)  # VOT = $20/hr
            else:
                # Transit less affected by departure time
                base_time = 40.0
                cost = base_time

            # Schedule delay cost
            preferred_arrival = 9  # 9 AM
            travel_time_hr = base_time / 60
            arrival = departure + travel_time_hr

            if arrival < preferred_arrival:
                # Early
                cost += (preferred_arrival - arrival) * 10  # $10/hr early
            else:
                # Late
                cost += (arrival - preferred_arrival) * 20  # $20/hr late

            costs.append(cost)

        return np.array(costs)

    def _simulate_dynamics(
        self,
        states: list[tuple[str, int]],
        initial_dist: np.ndarray,
        base_costs: np.ndarray,
        scenario: Scenario,
    ) -> pd.DataFrame:
        """Simulate day-to-day dynamics.

        Args:
            states: List of states
            initial_dist: Initial distribution
            base_costs: Costs per state
            scenario: Scenario

        Returns:
            DataFrame with daily distributions
        """
        n_states = len(states)
        distribution = initial_dist.copy()
        trajectory_data = []

        for day in range(self.n_days):
            # Compute costs with congestion effect
            # More people in auto -> higher cost
            auto_share = sum(
                distribution[i] for i, (m, d) in enumerate(states) if m == "auto"
            )
            congestion_factor = 1 + 0.5 * auto_share

            costs = base_costs.copy()
            for i, (mode, _) in enumerate(states):
                if mode == "auto":
                    costs[i] *= congestion_factor

            # Compute transition matrix
            P = self._compute_transition_matrix(costs)

            # Update distribution
            new_distribution = distribution @ P

            # Record state
            record = {"day": day}
            for i, state in enumerate(states):
                record[f"state_{i}"] = distribution[i]

            # Mode shares
            auto_share = sum(
                distribution[i] for i, (m, d) in enumerate(states) if m == "auto"
            )
            record["mode_share_auto"] = auto_share
            record["mode_share_transit"] = 1 - auto_share

            trajectory_data.append(record)
            distribution = new_distribution

        return pd.DataFrame(trajectory_data)

    def _compute_transition_matrix(self, costs: np.ndarray) -> np.ndarray:
        """Compute state transition matrix.

        P[i,j] = probability of transitioning from state i to state j

        Args:
            costs: Current costs per state

        Returns:
            Transition probability matrix
        """
        n_states = len(costs)
        P = np.zeros((n_states, n_states))

        # Logit probabilities for switching
        exp_costs = np.exp(-self.logit_scale * costs)
        logit_probs = exp_costs / exp_costs.sum()

        for i in range(n_states):
            # Probability of staying (inertia)
            P[i, i] = self.inertia

            # Probability of switching
            for j in range(n_states):
                if i != j:
                    P[i, j] = (1 - self.inertia) * logit_probs[j]

        return P

    def _distribution_to_aggregates(
        self,
        distribution: np.ndarray | pd.Series,
        states: list[tuple[str, int]],
    ) -> AggregateResults:
        """Convert state distribution to aggregate metrics.

        Args:
            distribution: Distribution over states
            states: List of states

        Returns:
            Aggregate results
        """
        if isinstance(distribution, pd.Series):
            distribution = distribution.values

        auto_share = sum(
            distribution[i] for i, (m, d) in enumerate(states) if m == "auto"
        )
        transit_share = 1 - auto_share

        return AggregateResults(
            mode_share_auto=auto_share,
            mode_share_transit=transit_share,
            mean_travel_time=25.0 * auto_share + 40.0 * transit_share,
            converged=True,
        )

    def find_stationary_distribution(
        self,
        states: list[tuple[str, int]],
        costs: np.ndarray,
    ) -> np.ndarray:
        """Find stationary distribution of Markov chain.

        Args:
            states: List of states
            costs: Costs per state

        Returns:
            Stationary distribution
        """
        P = self._compute_transition_matrix(costs)

        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))

        # Get stationary distribution
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        return stationary
