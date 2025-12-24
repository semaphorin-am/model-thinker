"""Static traffic assignment using Wardrop User Equilibrium.

This is the core traffic assignment model implementing the
Frank-Wolfe algorithm for finding user equilibrium.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.models.base import BaseModel, ModelRegistry
from congestion_pricing.network.graph import RoadNetwork, NetworkBuilder
from congestion_pricing.network.capacity import bpr_travel_time, bpr_integral
from congestion_pricing.policy.scenario import Scenario
from congestion_pricing.policy.results import (
    ModelResult,
    AggregateResults,
    LinkResults,
    ODResults,
)
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


@ModelRegistry.register("equilibrium")
class EquilibriumModel(BaseModel):
    """Static User Equilibrium traffic assignment model.

    Implements Wardrop User Equilibrium using the Frank-Wolfe algorithm.
    At equilibrium, all used paths between an OD pair have equal and
    minimum travel time.

    Config parameters:
        max_iterations: Maximum Frank-Wolfe iterations (default 100)
        convergence_gap: Relative gap threshold (default 0.01)
        alpha: BPR alpha parameter (default 0.15)
        beta: BPR beta parameter (default 4.0)
        vot: Value of time in $/hour (default 20.0)
    """

    name = "equilibrium"
    version = "1.0.0"
    description = "Static User Equilibrium (Wardrop) assignment"

    supports_tolls = True
    supports_transit = False
    supports_stochastic = False
    supports_dynamics = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.max_iterations = self.config.get("max_iterations", 100)
        self.convergence_gap = self.config.get("convergence_gap", 0.01)
        self.alpha = self.config.get("alpha", 0.15)
        self.beta = self.config.get("beta", 4.0)
        self.vot = self.config.get("vot", 20.0)

    def run(self, scenario: Scenario) -> ModelResult:
        """Run equilibrium assignment.

        Args:
            scenario: Scenario to evaluate

        Returns:
            ModelResult with flows and travel times
        """
        start_time = time.time()
        logger.info(f"Running equilibrium model for scenario: {scenario.name}")

        # Load or build network
        network = self._load_network(scenario)

        # Load demand
        od_demand = self._load_demand(scenario)

        # Get centroids
        centroids = network.get_centroid_nodes()
        if not centroids:
            # Use all nodes as potential OD points
            centroids = list(network.graph.nodes())[:50]

        # Run Frank-Wolfe assignment
        flows, times, gap, iterations = self._frank_wolfe(
            network, od_demand, centroids, scenario
        )

        runtime = time.time() - start_time

        # Compute aggregates
        aggregates = self._compute_aggregates(
            network, flows, times, od_demand, scenario
        )
        aggregates.runtime_seconds = runtime
        aggregates.iterations = iterations
        aggregates.relative_gap = gap
        aggregates.converged = gap < self.convergence_gap

        # Create link results
        link_df = self._create_link_results(network, flows, times)
        link_results = LinkResults(data=link_df)

        logger.info(
            f"Equilibrium converged in {iterations} iterations, "
            f"gap={gap:.4f}, runtime={runtime:.1f}s"
        )

        return self._create_result(
            scenario=scenario,
            aggregates=aggregates,
            link_results=link_results,
        )

    def _load_network(self, scenario: Scenario) -> RoadNetwork:
        """Load or build network from scenario.

        Args:
            scenario: Scenario with network path

        Returns:
            RoadNetwork object
        """
        from pathlib import Path

        network_path = Path(scenario.network_path)

        if (network_path / "graph.gpickle").exists():
            return RoadNetwork.load(network_path / "graph.gpickle")

        if (network_path / "nodes.parquet").exists():
            builder = NetworkBuilder()
            return builder.from_directory(network_path)

        # Create simple test network
        logger.warning("No network found, creating test network")
        return self._create_test_network()

    def _create_test_network(self) -> RoadNetwork:
        """Create a simple test network."""
        network = RoadNetwork()

        # Create grid network
        n = 5
        for i in range(n):
            for j in range(n):
                node_id = f"n_{i}_{j}"
                network.add_node(node_id, lat=51.5 + i * 0.01, lon=-0.1 + j * 0.01)

        for i in range(n):
            for j in range(n):
                if j < n - 1:
                    network.add_edge(
                        f"n_{i}_{j}", f"n_{i}_{j+1}",
                        length_m=1000, capacity_vph=1000, fft_minutes=2.0
                    )
                    network.add_edge(
                        f"n_{i}_{j+1}", f"n_{i}_{j}",
                        length_m=1000, capacity_vph=1000, fft_minutes=2.0
                    )
                if i < n - 1:
                    network.add_edge(
                        f"n_{i}_{j}", f"n_{i+1}_{j}",
                        length_m=1000, capacity_vph=1000, fft_minutes=2.0
                    )
                    network.add_edge(
                        f"n_{i+1}_{j}", f"n_{i}_{j}",
                        length_m=1000, capacity_vph=1000, fft_minutes=2.0
                    )

        return network

    def _load_demand(self, scenario: Scenario) -> dict[tuple[str, str], float]:
        """Load OD demand from scenario.

        Args:
            scenario: Scenario with demand

        Returns:
            Dictionary mapping (origin, dest) to demand
        """
        scenario.load()

        if scenario.od_matrix is not None:
            demand = {}
            for _, row in scenario.od_matrix.iterrows():
                key = (str(row["origin_zone"]), str(row["dest_zone"]))
                demand[key] = row.get("demand_auto", 0.0) * scenario.demand_multiplier
            return demand

        # Create synthetic demand
        return {("n_0_0", "n_4_4"): 500, ("n_0_4", "n_4_0"): 500}

    def _frank_wolfe(
        self,
        network: RoadNetwork,
        od_demand: dict[tuple[str, str], float],
        centroids: list[str],
        scenario: Scenario,
    ) -> tuple[dict, dict, float, int]:
        """Run Frank-Wolfe algorithm for user equilibrium.

        Args:
            network: Road network
            od_demand: OD demand matrix
            centroids: List of centroid nodes
            scenario: Scenario for toll info

        Returns:
            Tuple of (flows, travel_times, gap, iterations)
        """
        edges = list(network.graph.edges())
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        # Get edge attributes as arrays
        fft = np.array([network.graph[u][v].get("fft_minutes", 1.0) for u, v in edges])
        capacity = np.array([network.graph[u][v].get("capacity_vph", 1800) for u, v in edges])

        # Get toll costs
        toll_costs = self._get_toll_costs(network, edges, scenario)

        # Initialize flows to zero
        flows = np.zeros(n_edges)

        # All-or-nothing assignment for initial solution
        flows = self._all_or_nothing(
            network, od_demand, edges, edge_to_idx, fft, toll_costs
        )

        gap = float("inf")
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            # Update travel times
            times = bpr_travel_time(fft, flows, capacity, self.alpha, self.beta)

            # Add toll to generalized cost
            gen_cost = times + toll_costs / (self.vot / 60)

            # Update edge weights
            for i, (u, v) in enumerate(edges):
                network.graph[u][v]["travel_time"] = gen_cost[i]

            # All-or-nothing assignment with current costs
            target_flows = self._all_or_nothing(
                network, od_demand, edges, edge_to_idx, gen_cost, toll_costs
            )

            # Line search for optimal step size
            step_size = self._line_search(
                flows, target_flows, fft, capacity, toll_costs
            )

            # Update flows
            new_flows = flows + step_size * (target_flows - flows)

            # Compute convergence gap
            gap = self._compute_gap(flows, target_flows, times)

            flows = new_flows

            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: gap={gap:.6f}, step={step_size:.4f}")

            if gap < self.convergence_gap:
                break

        # Final travel times
        times = bpr_travel_time(fft, flows, capacity, self.alpha, self.beta)

        # Convert to dictionaries
        flow_dict = {edges[i]: flows[i] for i in range(n_edges)}
        time_dict = {edges[i]: times[i] for i in range(n_edges)}

        return flow_dict, time_dict, gap, iteration

    def _all_or_nothing(
        self,
        network: RoadNetwork,
        od_demand: dict[tuple[str, str], float],
        edges: list[tuple],
        edge_to_idx: dict,
        costs: np.ndarray,
        toll_costs: np.ndarray,
    ) -> np.ndarray:
        """Perform all-or-nothing assignment.

        Args:
            network: Road network
            od_demand: OD demand
            edges: List of edges
            edge_to_idx: Edge to index mapping
            costs: Current edge costs
            toll_costs: Toll costs

        Returns:
            Flow array from AON assignment
        """
        import networkx as nx

        flows = np.zeros(len(edges))

        # Update edge weights for shortest path
        for i, (u, v) in enumerate(edges):
            network.graph[u][v]["weight"] = costs[i]

        for (origin, dest), demand in od_demand.items():
            if demand <= 0:
                continue

            try:
                path = nx.shortest_path(network.graph, origin, dest, weight="weight")

                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in edge_to_idx:
                        flows[edge_to_idx[edge]] += demand

            except nx.NetworkXNoPath:
                continue

        return flows

    def _line_search(
        self,
        flows: np.ndarray,
        target: np.ndarray,
        fft: np.ndarray,
        capacity: np.ndarray,
        toll_costs: np.ndarray,
    ) -> float:
        """Bisection line search for optimal step size.

        Args:
            flows: Current flows
            target: Target flows from AON
            fft: Free-flow times
            capacity: Capacities
            toll_costs: Toll costs

        Returns:
            Optimal step size in [0, 1]
        """
        def objective(step):
            test_flows = flows + step * (target - flows)
            return np.sum(bpr_integral(fft, test_flows, capacity, self.alpha, self.beta))

        # Simple golden section search
        a, b = 0.0, 1.0
        phi = (1 + np.sqrt(5)) / 2

        for _ in range(20):
            c = b - (b - a) / phi
            d = a + (b - a) / phi

            if objective(c) < objective(d):
                b = d
            else:
                a = c

        return (a + b) / 2

    def _compute_gap(
        self,
        flows: np.ndarray,
        target: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """Compute relative gap for convergence check.

        Args:
            flows: Current flows
            target: Target flows
            times: Current travel times

        Returns:
            Relative gap
        """
        numerator = np.sum(times * (flows - target))
        denominator = np.sum(times * flows)

        if denominator < 1e-10:
            return 0.0

        return abs(numerator / denominator)

    def _get_toll_costs(
        self,
        network: RoadNetwork,
        edges: list[tuple],
        scenario: Scenario,
    ) -> np.ndarray:
        """Get toll costs for each edge.

        Args:
            network: Road network
            edges: List of edges
            scenario: Scenario with toll definition

        Returns:
            Array of toll costs per edge
        """
        tolls = np.zeros(len(edges))

        if scenario.toll is None:
            return tolls

        for i, (u, v) in enumerate(edges):
            if network.graph[u][v].get("in_toll_zone", False):
                # Use average toll rate
                avg_toll = np.mean(list(scenario.toll.rates.values()))
                tolls[i] = avg_toll

        return tolls

    def _compute_aggregates(
        self,
        network: RoadNetwork,
        flows: dict,
        times: dict,
        od_demand: dict,
        scenario: Scenario,
    ) -> AggregateResults:
        """Compute aggregate metrics from assignment results.

        Args:
            network: Road network
            flows: Link flows
            times: Link travel times
            od_demand: OD demand
            scenario: Scenario

        Returns:
            Aggregate results
        """
        # Total VKT and VHT
        total_vkt = 0.0
        total_vht = 0.0
        weighted_speed = 0.0

        for (u, v), flow in flows.items():
            length_km = network.graph[u][v].get("length_m", 1000) / 1000
            time_hr = times.get((u, v), 1.0) / 60

            total_vkt += flow * length_km
            total_vht += flow * time_hr

            if time_hr > 0:
                weighted_speed += flow * length_km / time_hr

        avg_speed = weighted_speed / total_vht if total_vht > 0 else 30.0

        # Mean travel time (flow-weighted)
        if total_vht > 0:
            mean_tt = total_vht / sum(flows.values()) * 60 if sum(flows.values()) > 0 else 25.0
        else:
            mean_tt = 25.0

        # Toll revenue
        toll_revenue = 0.0
        if scenario.toll is not None:
            avg_toll = np.mean(list(scenario.toll.rates.values()))
            for (u, v), flow in flows.items():
                if network.graph[u][v].get("in_toll_zone", False):
                    toll_revenue += flow * avg_toll

        # Emissions estimate
        emissions = total_vkt * 0.15  # 150 g CO2/km

        return AggregateResults(
            mean_travel_time=mean_tt,
            total_vkt=total_vkt,
            total_vht=total_vht,
            avg_speed_kmh=avg_speed,
            toll_revenue=toll_revenue,
            emissions_co2_kg=emissions,
            mode_share_auto=1.0,  # This model only handles auto
        )

    def _create_link_results(
        self,
        network: RoadNetwork,
        flows: dict,
        times: dict,
    ) -> pd.DataFrame:
        """Create link results DataFrame.

        Args:
            network: Road network
            flows: Link flows
            times: Link travel times

        Returns:
            DataFrame with link results
        """
        records = []
        for (u, v), flow in flows.items():
            length_km = network.graph[u][v].get("length_m", 1000) / 1000
            tt = times.get((u, v), 1.0)
            capacity = network.graph[u][v].get("capacity_vph", 1800)

            records.append({
                "edge_id": f"{u}_{v}",
                "from_node": u,
                "to_node": v,
                "period": "all_day",
                "flow_vph": flow,
                "travel_time_min": tt,
                "speed_kmh": length_km / (tt / 60) if tt > 0 else 0,
                "volume_capacity_ratio": flow / capacity if capacity > 0 else 0,
            })

        return pd.DataFrame(records)
