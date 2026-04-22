"""Game board: self-contained espionage network for Shadow Network.

The board is an nx.DiGraph where nodes are locations in a facility
and edges are infiltration passages. Detection risk on each edge
follows the BPR function: risk = base_risk * (1 + alpha * (agents/capacity)^beta).
"""

import networkx as nx
import numpy as np

from congestion_pricing.network.capacity import bpr_travel_time


FACILITY_NODES = {
    "staging": {"label": "Staging Area", "pos": (0, 3), "zone": "exterior"},
    "roof": {"label": "Rooftop Entry", "pos": (2, 5), "zone": "perimeter"},
    "loading": {"label": "Loading Dock", "pos": (2, 1), "zone": "perimeter"},
    "vent_shaft": {"label": "Ventilation Shaft", "pos": (4, 5), "zone": "interior"},
    "main_hall": {"label": "Main Hallway", "pos": (4, 3), "zone": "interior"},
    "basement": {"label": "Basement Tunnel", "pos": (4, 1), "zone": "interior"},
    "lab": {"label": "Research Lab", "pos": (6, 4), "zone": "secure"},
    "comms": {"label": "Comms Room", "pos": (6, 2), "zone": "secure"},
    "server": {"label": "Server Room", "pos": (8, 4), "zone": "vault"},
    "vault": {"label": "The Vault", "pos": (8, 2), "zone": "vault"},
}

FACILITY_EDGES = [
    ("staging", "roof", {"base_risk": 3.0, "capacity": 80, "label": "Fire Escape"}),
    ("staging", "loading", {"base_risk": 4.0, "capacity": 120, "label": "Freight Entrance"}),
    ("staging", "main_hall", {"base_risk": 6.0, "capacity": 200, "label": "Front Door"}),
    ("roof", "vent_shaft", {"base_risk": 5.0, "capacity": 40, "label": "Roof Hatch"}),
    ("roof", "main_hall", {"base_risk": 4.0, "capacity": 60, "label": "Stairwell"}),
    ("loading", "basement", {"base_risk": 3.0, "capacity": 100, "label": "Cargo Lift"}),
    ("loading", "main_hall", {"base_risk": 5.0, "capacity": 80, "label": "Service Corridor"}),
    ("vent_shaft", "lab", {"base_risk": 2.0, "capacity": 30, "label": "Air Duct"}),
    ("vent_shaft", "comms", {"base_risk": 4.0, "capacity": 50, "label": "Ceiling Access"}),
    ("main_hall", "lab", {"base_risk": 7.0, "capacity": 150, "label": "Main Corridor"}),
    ("main_hall", "comms", {"base_risk": 6.0, "capacity": 150, "label": "East Wing"}),
    ("basement", "comms", {"base_risk": 3.0, "capacity": 60, "label": "Utility Tunnel"}),
    ("basement", "vault", {"base_risk": 8.0, "capacity": 40, "label": "Sub-basement"}),
    ("lab", "server", {"base_risk": 4.0, "capacity": 80, "label": "Secure Bridge"}),
    ("comms", "server", {"base_risk": 5.0, "capacity": 60, "label": "Network Conduit"}),
    ("comms", "vault", {"base_risk": 6.0, "capacity": 70, "label": "Inner Corridor"}),
    ("server", "vault", {"base_risk": 3.0, "capacity": 50, "label": "Server Passage"}),
    ("lab", "vault", {"base_risk": 9.0, "capacity": 100, "label": "Direct Access"}),
]

ZONE_COLORS = {
    "exterior": "#4a90d9",
    "perimeter": "#7bc47f",
    "interior": "#f5a623",
    "secure": "#d94a4a",
    "vault": "#9b59b6",
}


def create_board() -> nx.DiGraph:
    """Create the espionage facility network.

    Returns:
        Directed graph with node/edge attributes for the game board.
    """
    G = nx.DiGraph()

    for node_id, attrs in FACILITY_NODES.items():
        G.add_node(node_id, **attrs)

    for src, dst, attrs in FACILITY_EDGES:
        G.add_edge(src, dst, **attrs)

    return G


def compute_detection_risk(
    base_risk: float | np.ndarray,
    agent_count: float | np.ndarray,
    capacity: float | np.ndarray,
    patrol_intensity: float | np.ndarray = 0.0,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float | np.ndarray:
    """Compute detection risk on an edge using BPR function.

    This is the congestion pricing BPR function re-interpreted:
    - free_flow_time → base detection risk (minimum risk when passage is empty)
    - flow → agent count on this passage
    - capacity → how many agents before suspicion spikes
    - toll → patrol intensity (additional fixed cost from security presence)

    Args:
        base_risk: Minimum detection risk (= free-flow time)
        agent_count: Number of agents using this passage (= flow)
        capacity: Agent throughput before suspicion spikes (= link capacity)
        patrol_intensity: Security patrol cost added to this edge (= toll)
        alpha: BPR alpha (default 0.15)
        beta: BPR beta (default 4.0)

    Returns:
        Total detection risk (in same units as base_risk)
    """
    congestion_risk = bpr_travel_time(base_risk, agent_count, capacity, alpha, beta)
    return congestion_risk + patrol_intensity


def get_all_routes(G: nx.DiGraph, origin: str = "staging", targets: list[str] | None = None) -> list[list[str]]:
    """Find all simple paths from origin to target nodes.

    Args:
        G: The game board graph
        origin: Starting node
        targets: Target nodes (default: ["server", "vault"])

    Returns:
        List of paths (each path is a list of node IDs)
    """
    if targets is None:
        targets = ["server", "vault"]

    routes = []
    for target in targets:
        for path in nx.all_simple_paths(G, origin, target, cutoff=6):
            routes.append(path)

    return routes


def route_base_risk(G: nx.DiGraph, route: list[str]) -> float:
    """Compute the total base risk of a route (sum of edge base risks)."""
    total = 0.0
    for i in range(len(route) - 1):
        edge = G[route[i]][route[i + 1]]
        total += edge["base_risk"]
    return total


def route_bottleneck_capacity(G: nx.DiGraph, route: list[str]) -> float:
    """Find the minimum capacity edge on a route (the bottleneck)."""
    caps = []
    for i in range(len(route) - 1):
        edge = G[route[i]][route[i + 1]]
        caps.append(edge["capacity"])
    return min(caps) if caps else 0.0
