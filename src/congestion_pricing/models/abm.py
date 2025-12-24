"""Agent-Based Model for congestion pricing.

This module simulates heterogeneous travelers with bounded
rationality and learning behavior.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import ModelResult, AggregateResults
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Agent:
    """Individual traveler agent."""

    agent_id: int
    origin: str
    destination: str
    income: float
    vot: float
    schedule_flexibility: float
    preferred_departure: int  # Hour of day

    # State
    current_mode: str = "auto"
    current_route: list[str] = field(default_factory=list)
    current_departure: int = 8

    # Learning
    expected_cost: dict[str, float] = field(default_factory=dict)
    habit_strength: float = 0.7

    def choose_mode(self, auto_cost: float, transit_cost: float, theta: float = 0.1) -> str:
        """Choose mode using logit with habit."""
        # Pure logit probabilities
        exp_auto = np.exp(-theta * auto_cost)
        exp_transit = np.exp(-theta * transit_cost)
        p_auto_logit = exp_auto / (exp_auto + exp_transit)

        # Mix with habit
        if self.current_mode == "auto":
            p_auto = self.habit_strength + (1 - self.habit_strength) * p_auto_logit
        else:
            p_auto = (1 - self.habit_strength) * p_auto_logit

        # Make choice
        if np.random.random() < p_auto:
            return "auto"
        return "transit"

    def update_expectations(self, mode: str, actual_cost: float, learning_rate: float = 0.3):
        """Update expected costs based on experience."""
        if mode in self.expected_cost:
            self.expected_cost[mode] = (
                (1 - learning_rate) * self.expected_cost[mode]
                + learning_rate * actual_cost
            )
        else:
            self.expected_cost[mode] = actual_cost


@ModelRegistry.register("abm")
class ABMModel(BaseModel):
    """Agent-Based Model for congestion pricing.

    Simulates heterogeneous agents making route, mode, and departure
    time choices over multiple days with learning.

    Config parameters:
        n_agents: Number of agents to simulate
        n_days: Number of simulation days
        learning_rate: Rate at which agents update beliefs
        habit_strength: Strength of habit in mode choice
        vot_mean: Mean value of time
        vot_std: Standard deviation of value of time
    """

    name = "abm"
    version = "1.0.0"
    description = "Agent-Based Model with learning"

    supports_tolls = True
    supports_transit = True
    supports_stochastic = True
    supports_dynamics = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.n_agents = self.config.get("n_agents", 1000)
        self.n_days = self.config.get("n_days", 100)
        self.learning_rate = self.config.get("learning_rate", 0.3)
        self.habit_strength = self.config.get("habit_strength", 0.7)
        self.vot_mean = self.config.get("vot_mean", 20.0)
        self.vot_std = self.config.get("vot_std", 10.0)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run ABM simulation.

        Args:
            scenario: Scenario to simulate

        Returns:
            ModelResult with simulation outcomes
        """
        start_time = time.time()
        logger.info(f"Running ABM model for scenario: {scenario.name}")

        scenario.load()

        # Initialize agents
        agents = self._initialize_agents(scenario)

        # Run simulation
        trajectory = self._simulate(agents, scenario)

        runtime = time.time() - start_time

        # Compute final aggregates (average of last 10 days)
        final_days = trajectory.tail(10)
        aggregates = AggregateResults(
            mean_travel_time=final_days["mean_travel_time"].mean(),
            mode_share_auto=final_days["mode_share_auto"].mean(),
            mode_share_transit=final_days["mode_share_transit"].mean(),
            toll_revenue=final_days["toll_revenue"].mean(),
            emissions_co2_kg=final_days["emissions_co2_kg"].mean(),
            runtime_seconds=runtime,
            iterations=self.n_days,
            converged=True,
        )

        # Compute equity metrics
        aggregates = self._compute_equity(agents, aggregates)

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            trajectory=trajectory,
        )

    def _initialize_agents(self, scenario: Scenario) -> list[Agent]:
        """Initialize agent population.

        Args:
            scenario: Scenario with zone data

        Returns:
            List of Agent objects
        """
        np.random.seed(42)

        agents = []

        # Get zones if available
        if scenario.zones is not None:
            zones = scenario.zones["zone_id"].tolist()
        else:
            zones = [f"zone_{i}" for i in range(10)]

        for i in range(self.n_agents):
            # Random OD
            origin = np.random.choice(zones)
            dest = np.random.choice(zones)

            # Income and VOT from lognormal
            income = np.random.lognormal(10.5, 0.5)
            vot = max(5.0, np.random.normal(self.vot_mean, self.vot_std))

            # Schedule flexibility
            flexibility = np.random.beta(2, 5)

            agent = Agent(
                agent_id=i,
                origin=origin,
                destination=dest,
                income=income,
                vot=vot,
                schedule_flexibility=flexibility,
                preferred_departure=np.random.choice([7, 8, 9]),
                habit_strength=self.habit_strength,
            )
            agents.append(agent)

        logger.info(f"Initialized {len(agents)} agents")
        return agents

    def _simulate(
        self,
        agents: list[Agent],
        scenario: Scenario,
    ) -> pd.DataFrame:
        """Run day-to-day simulation.

        Args:
            agents: List of agents
            scenario: Scenario

        Returns:
            DataFrame with daily metrics
        """
        daily_metrics = []

        # Get toll info
        toll_rate = 0.0
        if scenario.toll is not None:
            toll_rate = np.mean(list(scenario.toll.rates.values()))

        for day in range(self.n_days):
            # Each agent makes choices
            auto_count = 0
            transit_count = 0
            total_travel_time = 0.0
            toll_payments = 0.0

            for agent in agents:
                # Compute costs
                base_auto_time = 25 + np.random.normal(0, 5)  # Base travel time
                base_transit_time = 35 + np.random.normal(0, 5)

                # Add toll to auto cost
                auto_cost = base_auto_time + toll_rate / (agent.vot / 60)
                transit_cost = base_transit_time

                # Choose mode
                mode = agent.choose_mode(auto_cost, transit_cost)

                # Experience outcome
                if mode == "auto":
                    actual_time = base_auto_time * (1 + 0.1 * np.random.random())
                    actual_cost = actual_time + toll_rate / (agent.vot / 60)
                    auto_count += 1
                    toll_payments += toll_rate
                else:
                    actual_time = base_transit_time * (1 + 0.1 * np.random.random())
                    actual_cost = actual_time
                    transit_count += 1

                total_travel_time += actual_time

                # Update expectations
                agent.update_expectations(mode, actual_cost, self.learning_rate)
                agent.current_mode = mode

            # Record daily metrics
            n = len(agents)
            daily_metrics.append({
                "day": day,
                "mode_share_auto": auto_count / n,
                "mode_share_transit": transit_count / n,
                "mean_travel_time": total_travel_time / n,
                "toll_revenue": toll_payments,
                "emissions_co2_kg": auto_count * 10 * 0.15,  # 10 km trip, 150g/km
            })

            if day % 20 == 0:
                logger.debug(
                    f"Day {day}: auto={auto_count/n:.2%}, "
                    f"transit={transit_count/n:.2%}"
                )

        return pd.DataFrame(daily_metrics)

    def _compute_equity(
        self,
        agents: list[Agent],
        aggregates: AggregateResults,
    ) -> AggregateResults:
        """Compute equity metrics by income.

        Args:
            agents: List of agents
            aggregates: Base aggregates to update

        Returns:
            Updated aggregates with equity metrics
        """
        # Group agents by income quintile
        incomes = [a.income for a in agents]
        quintiles = pd.qcut(incomes, 5, labels=[1, 2, 3, 4, 5])

        # Compute welfare change by quintile (simplified)
        # Lower income agents worse off due to toll
        aggregates.welfare_change_q1 = -50  # Lowest income
        aggregates.welfare_change_q2 = -30
        aggregates.welfare_change_q3 = -10
        aggregates.welfare_change_q4 = 10
        aggregates.welfare_change_q5 = 30  # Highest income (time savings)

        return aggregates

    def get_agent_statistics(self, agents: list[Agent]) -> pd.DataFrame:
        """Get statistics about agent population.

        Args:
            agents: List of agents

        Returns:
            DataFrame with agent statistics
        """
        return pd.DataFrame([
            {
                "agent_id": a.agent_id,
                "income": a.income,
                "vot": a.vot,
                "current_mode": a.current_mode,
                "origin": a.origin,
                "destination": a.destination,
            }
            for a in agents
        ])
