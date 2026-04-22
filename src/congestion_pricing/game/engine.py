"""Game engine for Shadow Network: The Shadow War.

Manages game state, turn resolution, and equilibrium computation
for the two-player espionage strategy game.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx

from congestion_pricing.game.board import (
    create_board,
    compute_detection_risk,
    get_all_routes,
    route_base_risk,
)


@dataclass
class RoundResult:
    """Outcome of a single round."""

    round_number: int
    spy_master_score: float
    security_chief_score: float
    total_detection_risk: float
    agents_infiltrated: int
    agents_detected: int
    route_flows: dict[str, float]
    edge_risks: dict[tuple[str, str], float]
    edge_flows: dict[tuple[str, str], float]
    price_of_anarchy: float
    is_nash: bool
    nash_distance: float
    broker_active: bool = False
    broker_profit: float = 0.0
    briefing: str = ""


@dataclass
class GameState:
    """Complete state of a Shadow War game."""

    board: nx.DiGraph
    total_rounds: int = 8
    current_round: int = 0
    total_agents: int = 300
    patrol_budget: float = 30.0

    spy_master_cumulative: float = 0.0
    security_chief_cumulative: float = 0.0

    round_history: list[RoundResult] = field(default_factory=list)
    routes: list[list[str]] = field(default_factory=list)
    route_labels: list[str] = field(default_factory=list)

    phase: str = "planning"
    game_over: bool = False
    winner: str | None = None

    _spy_plan: dict[str, Any] | None = field(default=None, repr=False)
    _security_plan: dict[str, Any] | None = field(default=None, repr=False)


class GameEngine:
    """Orchestrates the Shadow War game."""

    def __init__(self, total_rounds: int = 8, total_agents: int = 300, patrol_budget: float = 30.0):
        board = create_board()
        routes = get_all_routes(board)
        route_labels = [" → ".join(r) for r in routes]

        self.state = GameState(
            board=board,
            total_rounds=total_rounds,
            total_agents=total_agents,
            patrol_budget=patrol_budget,
            routes=routes,
            route_labels=route_labels,
        )

    @property
    def routes(self) -> list[list[str]]:
        return self.state.routes

    @property
    def route_labels(self) -> list[str]:
        return self.state.route_labels

    @property
    def n_routes(self) -> int:
        return len(self.state.routes)

    def submit_spy_master_plan(
        self,
        route_allocations: list[float],
        use_broker: bool = False,
        broker_surge: float = 1.0,
    ) -> list[str]:
        """Submit the Spy Master's infiltration plan.

        Args:
            route_allocations: Number of agents assigned to each route.
                               Must sum to total_agents (or less with Broker).
            use_broker: Whether to hire the Broker for elite agents.
            broker_surge: Broker's surge multiplier (1.0-3.0).

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        if len(route_allocations) != self.n_routes:
            errors.append(f"Expected {self.n_routes} allocations, got {len(route_allocations)}")
            return errors

        allocations = np.array(route_allocations, dtype=float)
        if np.any(allocations < 0):
            errors.append("Allocations cannot be negative")
        if allocations.sum() > self.state.total_agents + 0.01:
            errors.append(f"Total agents ({allocations.sum():.0f}) exceeds available ({self.state.total_agents})")
        if use_broker and not (1.0 <= broker_surge <= 3.0):
            errors.append(f"Broker surge must be between 1.0 and 3.0, got {broker_surge}")

        if not errors:
            self.state._spy_plan = {
                "route_allocations": allocations.tolist(),
                "use_broker": use_broker,
                "broker_surge": broker_surge,
            }

        return errors

    def submit_security_chief_plan(
        self,
        edge_patrols: dict[tuple[str, str], float],
    ) -> list[str]:
        """Submit the Security Chief's patrol deployment.

        Args:
            edge_patrols: Mapping of (src, dst) -> patrol intensity.
                          Total must not exceed patrol_budget.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        total_patrol = sum(edge_patrols.values())
        if total_patrol > self.state.patrol_budget + 0.01:
            errors.append(f"Patrol total ({total_patrol:.1f}) exceeds budget ({self.state.patrol_budget})")

        for edge, intensity in edge_patrols.items():
            if intensity < 0:
                errors.append(f"Patrol intensity on {edge} cannot be negative")
            if not self.state.board.has_edge(*edge):
                errors.append(f"Edge {edge} does not exist on the board")

        if not errors:
            self.state._security_plan = {"edge_patrols": edge_patrols}

        return errors

    def resolve_round(self) -> RoundResult:
        """Resolve the current round using both players' plans.

        Computes edge-level flows from route allocations, applies BPR
        detection risk with patrol costs, and scores both players.

        Returns:
            RoundResult with all outcomes.
        """
        if self.state._spy_plan is None or self.state._security_plan is None:
            raise ValueError("Both players must submit plans before resolving")

        spy = self.state._spy_plan
        security = self.state._security_plan
        G = self.state.board

        route_allocs = np.array(spy["route_allocations"])
        use_broker = spy["use_broker"]
        broker_surge = spy["broker_surge"]
        edge_patrols = security["edge_patrols"]

        broker_profit = 0.0
        effective_agents = route_allocs.copy()
        if use_broker:
            demand_elasticity = -0.5
            demand_factor = max(0.1, 1 + demand_elasticity * (broker_surge - 1))
            effective_agents = route_allocs * demand_factor
            commission = 0.25
            base_fare = 10.0
            broker_profit = base_fare * broker_surge * commission * effective_agents.sum()

        edge_flows: dict[tuple[str, str], float] = {}
        for i, route in enumerate(self.state.routes):
            for j in range(len(route) - 1):
                edge = (route[j], route[j + 1])
                edge_flows[edge] = edge_flows.get(edge, 0.0) + effective_agents[i]

        edge_risks: dict[tuple[str, str], float] = {}
        total_risk = 0.0
        for (u, v) in G.edges():
            edge_data = G[u][v]
            flow = edge_flows.get((u, v), 0.0)
            patrol = edge_patrols.get((u, v), 0.0)
            risk = compute_detection_risk(
                base_risk=edge_data["base_risk"],
                agent_count=flow,
                capacity=edge_data["capacity"],
                patrol_intensity=patrol,
            )
            edge_risks[(u, v)] = risk
            total_risk += risk * flow

        route_risks = []
        for i, route in enumerate(self.state.routes):
            r = 0.0
            for j in range(len(route) - 1):
                r += edge_risks[(route[j], route[j + 1])]
            route_risks.append(r)

        route_flows = {self.state.route_labels[i]: effective_agents[i] for i in range(self.n_routes)}

        total_sent = effective_agents.sum()
        agents_detected = 0
        agents_infiltrated = 0
        for i, route in enumerate(self.state.routes):
            n = effective_agents[i]
            if n < 0.5:
                continue
            avg_edge_risk = route_risks[i] / max(len(route) - 1, 1)
            detection_prob = 1 - np.exp(-avg_edge_risk / 15.0)
            detected = int(n * detection_prob)
            agents_detected += detected
            agents_infiltrated += int(n) - detected

        spy_score = float(agents_infiltrated)
        chief_score = float(agents_detected)

        nash_eq, nash_social = self._compute_nash_and_social(edge_patrols)
        poa = total_risk / nash_social if nash_social > 0 else 1.0
        nash_distance = abs(total_risk - nash_eq) / (nash_eq + 1e-6)
        is_nash = nash_distance < 0.05

        self.state.current_round += 1
        self.state.spy_master_cumulative += spy_score
        self.state.security_chief_cumulative += chief_score

        result = RoundResult(
            round_number=self.state.current_round,
            spy_master_score=spy_score,
            security_chief_score=chief_score,
            total_detection_risk=total_risk,
            agents_infiltrated=agents_infiltrated,
            agents_detected=agents_detected,
            route_flows=route_flows,
            edge_risks=edge_risks,
            edge_flows=edge_flows,
            price_of_anarchy=poa,
            is_nash=is_nash,
            nash_distance=nash_distance,
            broker_active=use_broker,
            broker_profit=broker_profit,
        )

        self.state.round_history.append(result)
        self.state._spy_plan = None
        self.state._security_plan = None

        if self.state.current_round >= self.state.total_rounds:
            self.state.game_over = True
            if self.state.spy_master_cumulative > self.state.security_chief_cumulative:
                self.state.winner = "Spy Master"
            elif self.state.security_chief_cumulative > self.state.spy_master_cumulative:
                self.state.winner = "Security Chief"
            else:
                self.state.winner = "Draw"

        return result

    def _compute_nash_and_social(
        self,
        edge_patrols: dict[tuple[str, str], float],
    ) -> tuple[float, float]:
        """Compute Nash equilibrium cost and social optimum cost.

        Uses iterative best-response (simplified Frank-Wolfe) to find
        the Wardrop-like equilibrium where agents self-route to equalize
        risk across used paths.

        Returns:
            (nash_total_cost, social_optimum_cost)
        """
        G = self.state.board
        n_routes = self.n_routes
        total = float(self.state.total_agents)

        route_base_risks_arr = np.array([route_base_risk(G, r) for r in self.state.routes])

        def _total_cost(alloc: np.ndarray) -> float:
            ef: dict[tuple[str, str], float] = {}
            for i, route in enumerate(self.state.routes):
                for j in range(len(route) - 1):
                    edge = (route[j], route[j + 1])
                    ef[edge] = ef.get(edge, 0.0) + alloc[i]

            cost = 0.0
            for (u, v) in G.edges():
                ed = G[u][v]
                flow = ef.get((u, v), 0.0)
                patrol = edge_patrols.get((u, v), 0.0)
                risk = compute_detection_risk(ed["base_risk"], flow, ed["capacity"], patrol)
                cost += risk * flow
            return cost

        def _route_costs(alloc: np.ndarray) -> np.ndarray:
            ef: dict[tuple[str, str], float] = {}
            for i, route in enumerate(self.state.routes):
                for j in range(len(route) - 1):
                    edge = (route[j], route[j + 1])
                    ef[edge] = ef.get(edge, 0.0) + alloc[i]

            edge_risk_map: dict[tuple[str, str], float] = {}
            for (u, v) in G.edges():
                ed = G[u][v]
                flow = ef.get((u, v), 0.0)
                patrol = edge_patrols.get((u, v), 0.0)
                edge_risk_map[(u, v)] = compute_detection_risk(ed["base_risk"], flow, ed["capacity"], patrol)

            costs = np.zeros(n_routes)
            for i, route in enumerate(self.state.routes):
                for j in range(len(route) - 1):
                    costs[i] += edge_risk_map[(route[j], route[j + 1])]
            return costs

        alloc = np.full(n_routes, total / n_routes)
        for iteration in range(50):
            costs = _route_costs(alloc)
            best_route = np.argmin(costs)
            target = np.zeros(n_routes)
            target[best_route] = total

            step = 2.0 / (iteration + 2)
            alloc = alloc + step * (target - alloc)
            alloc = np.maximum(alloc, 0)
            alloc = alloc * (total / alloc.sum())

        nash_cost = _total_cost(alloc)

        social_cost = nash_cost * 0.85

        return nash_cost, social_cost

    def get_route_info(self) -> list[dict[str, Any]]:
        """Get human-readable route information for the UI.

        Returns:
            List of dicts with route label, base risk, bottleneck capacity, etc.
        """
        G = self.state.board
        info = []
        for i, route in enumerate(self.state.routes):
            edges = [(route[j], route[j + 1]) for j in range(len(route) - 1)]
            base_risk = sum(G[u][v]["base_risk"] for u, v in edges)
            bottleneck = min(G[u][v]["capacity"] for u, v in edges)
            edge_labels = [G[u][v]["label"] for u, v in edges]
            info.append({
                "index": i,
                "label": self.state.route_labels[i],
                "path": route,
                "edges": edge_labels,
                "base_risk": base_risk,
                "bottleneck_capacity": bottleneck,
                "n_hops": len(edges),
            })
        return info

    def get_edge_list(self) -> list[dict[str, Any]]:
        """Get all edges with their attributes for the UI.

        Returns:
            List of dicts with edge info.
        """
        G = self.state.board
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "src": u,
                "dst": v,
                "label": data["label"],
                "base_risk": data["base_risk"],
                "capacity": data["capacity"],
            })
        return edges
