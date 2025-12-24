"""Network graph construction and manipulation.

This module provides the RoadNetwork class which wraps NetworkX
graphs with transportation-specific functionality.
"""

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


class RoadNetwork:
    """Transportation network wrapper around NetworkX graph.

    This class provides transportation-specific methods for
    network analysis and manipulation.
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        """Initialize network.

        Args:
            graph: Optional existing NetworkX graph
        """
        self.graph = graph or nx.DiGraph()
        self._shortest_paths_cache: dict[tuple, list] = {}

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.graph.number_of_edges()

    def add_node(
        self,
        node_id: str,
        lat: float,
        lon: float,
        is_centroid: bool = False,
        **attrs: Any,
    ) -> None:
        """Add a node to the network.

        Args:
            node_id: Unique node identifier
            lat: Latitude
            lon: Longitude
            is_centroid: Whether this is a zone centroid
            **attrs: Additional node attributes
        """
        self.graph.add_node(
            node_id,
            lat=lat,
            lon=lon,
            is_centroid=is_centroid,
            **attrs,
        )

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        length_m: float,
        capacity_vph: float,
        fft_minutes: float,
        **attrs: Any,
    ) -> None:
        """Add an edge to the network.

        Args:
            from_node: Origin node ID
            to_node: Destination node ID
            length_m: Length in meters
            capacity_vph: Capacity in vehicles per hour
            fft_minutes: Free-flow travel time in minutes
            **attrs: Additional edge attributes
        """
        self.graph.add_edge(
            from_node,
            to_node,
            length_m=length_m,
            capacity_vph=capacity_vph,
            fft_minutes=fft_minutes,
            flow=0.0,
            travel_time=fft_minutes,
            **attrs,
        )

    def get_edge_attribute(self, attr: str) -> dict[tuple[str, str], Any]:
        """Get attribute for all edges.

        Args:
            attr: Attribute name

        Returns:
            Dictionary mapping (from, to) to attribute value
        """
        return nx.get_edge_attributes(self.graph, attr)

    def set_edge_flows(self, flows: dict[tuple[str, str], float]) -> None:
        """Set flow values for edges.

        Args:
            flows: Dictionary mapping (from, to) to flow value
        """
        for (u, v), flow in flows.items():
            if self.graph.has_edge(u, v):
                self.graph[u][v]["flow"] = flow

    def set_edge_travel_times(self, times: dict[tuple[str, str], float]) -> None:
        """Set travel time values for edges.

        Args:
            times: Dictionary mapping (from, to) to travel time
        """
        for (u, v), time in times.items():
            if self.graph.has_edge(u, v):
                self.graph[u][v]["travel_time"] = time

    def compute_bpr_times(self, alpha: float = 0.15, beta: float = 4.0) -> None:
        """Compute travel times using BPR formula.

        t = t0 * (1 + alpha * (flow / capacity)^beta)

        Args:
            alpha: BPR alpha parameter
            beta: BPR beta parameter
        """
        for u, v, data in self.graph.edges(data=True):
            fft = data.get("fft_minutes", 1.0)
            flow = data.get("flow", 0.0)
            capacity = data.get("capacity_vph", 1800.0)

            vc_ratio = flow / capacity if capacity > 0 else 0
            travel_time = fft * (1 + alpha * (vc_ratio ** beta))

            self.graph[u][v]["travel_time"] = travel_time
            self.graph[u][v]["volume_capacity_ratio"] = vc_ratio

    def shortest_path(
        self,
        origin: str,
        destination: str,
        weight: str = "travel_time",
    ) -> list[str]:
        """Find shortest path between two nodes.

        Args:
            origin: Origin node ID
            destination: Destination node ID
            weight: Edge attribute to use as weight

        Returns:
            List of node IDs in path
        """
        try:
            return nx.shortest_path(self.graph, origin, destination, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def shortest_path_length(
        self,
        origin: str,
        destination: str,
        weight: str = "travel_time",
    ) -> float:
        """Find shortest path length between two nodes.

        Args:
            origin: Origin node ID
            destination: Destination node ID
            weight: Edge attribute to use as weight

        Returns:
            Path length (travel time)
        """
        try:
            return nx.shortest_path_length(self.graph, origin, destination, weight=weight)
        except nx.NetworkXNoPath:
            return float("inf")

    def all_pairs_shortest_paths(
        self,
        origins: list[str],
        destinations: list[str],
        weight: str = "travel_time",
    ) -> pd.DataFrame:
        """Compute shortest paths for all OD pairs.

        Args:
            origins: List of origin node IDs
            destinations: List of destination node IDs
            weight: Edge attribute to use as weight

        Returns:
            DataFrame with columns origin, destination, travel_time, path
        """
        records = []

        for origin in origins:
            try:
                lengths, paths = nx.single_source_dijkstra(
                    self.graph, origin, weight=weight
                )
                for dest in destinations:
                    if dest in lengths:
                        records.append({
                            "origin": origin,
                            "destination": dest,
                            "travel_time": lengths[dest],
                            "path": paths[dest],
                        })
            except nx.NetworkXError:
                continue

        return pd.DataFrame(records)

    def get_centroid_nodes(self) -> list[str]:
        """Get all centroid nodes.

        Returns:
            List of centroid node IDs
        """
        return [
            n for n, data in self.graph.nodes(data=True)
            if data.get("is_centroid", False)
        ]

    def mark_toll_zone(self, zone_polygon: list[tuple[float, float]]) -> int:
        """Mark edges inside a toll zone.

        Args:
            zone_polygon: List of (lon, lat) coordinates defining zone

        Returns:
            Number of edges marked
        """
        from congestion_pricing.utils.geo import point_in_polygon

        count = 0
        for u, v, data in self.graph.edges(data=True):
            # Use midpoint of edge
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]

            mid_lon = (u_data.get("lon", 0) + v_data.get("lon", 0)) / 2
            mid_lat = (u_data.get("lat", 0) + v_data.get("lat", 0)) / 2

            if point_in_polygon((mid_lon, mid_lat), zone_polygon):
                self.graph[u][v]["in_toll_zone"] = True
                count += 1

        logger.info(f"Marked {count} edges in toll zone")
        return count

    def to_edges_dataframe(self) -> pd.DataFrame:
        """Convert edges to DataFrame.

        Returns:
            DataFrame with edge attributes
        """
        records = []
        for u, v, data in self.graph.edges(data=True):
            record = {
                "from_node": u,
                "to_node": v,
                **data,
            }
            records.append(record)
        return pd.DataFrame(records)

    def to_nodes_dataframe(self) -> pd.DataFrame:
        """Convert nodes to DataFrame.

        Returns:
            DataFrame with node attributes
        """
        records = []
        for n, data in self.graph.nodes(data=True):
            record = {"node_id": n, **data}
            records.append(record)
        return pd.DataFrame(records)

    def save(self, path: str | Path) -> None:
        """Save network to file.

        Args:
            path: Output path (gpickle format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(self.graph, path)
        logger.info(f"Saved network to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RoadNetwork":
        """Load network from file.

        Args:
            path: Path to gpickle file

        Returns:
            Loaded RoadNetwork
        """
        graph = nx.read_gpickle(path)
        return cls(graph)

    def get_link_incidence_matrix(
        self,
        paths: dict[tuple[str, str], list[str]],
    ) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
        """Create link-path incidence matrix.

        Args:
            paths: Dictionary mapping (origin, dest) to path (list of nodes)

        Returns:
            Tuple of (matrix, path_list, edge_list)
        """
        edge_list = list(self.graph.edges())
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}

        path_list = list(paths.keys())

        # Create sparse incidence matrix
        n_paths = len(path_list)
        n_edges = len(edge_list)

        matrix = np.zeros((n_edges, n_paths))

        for j, od in enumerate(path_list):
            path = paths[od]
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in edge_to_idx:
                    matrix[edge_to_idx[edge], j] = 1

        return matrix, path_list, edge_list


class NetworkBuilder:
    """Builder class for constructing networks from data."""

    def __init__(self):
        """Initialize builder."""
        self.network = RoadNetwork()

    def from_dataframes(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> RoadNetwork:
        """Build network from DataFrames.

        Args:
            nodes_df: Nodes DataFrame
            edges_df: Edges DataFrame

        Returns:
            Constructed RoadNetwork
        """
        # Add nodes
        for _, row in nodes_df.iterrows():
            self.network.add_node(
                node_id=str(row["node_id"]),
                lat=row["lat"],
                lon=row["lon"],
                is_centroid=row.get("is_centroid", False),
            )

        # Add edges
        for _, row in edges_df.iterrows():
            self.network.add_edge(
                from_node=str(row["from_node"]),
                to_node=str(row["to_node"]),
                length_m=row["length_m"],
                capacity_vph=row.get("capacity_vph", 1800.0),
                fft_minutes=row.get("fft_minutes", row["length_m"] / 1000 / 50 * 60),
                road_class=row.get("road_class", "unclassified"),
                in_toll_zone=row.get("in_toll_zone", False),
            )

        logger.info(
            f"Built network with {self.network.n_nodes} nodes "
            f"and {self.network.n_edges} edges"
        )

        return self.network

    def from_directory(self, path: str | Path) -> RoadNetwork:
        """Load network from processed data directory.

        Args:
            path: Path to network directory containing nodes.parquet and edges.parquet

        Returns:
            Constructed RoadNetwork
        """
        path = Path(path)

        nodes_df = pd.read_parquet(path / "nodes.parquet")
        edges_df = pd.read_parquet(path / "edges.parquet")

        return self.from_dataframes(nodes_df, edges_df)

    def add_zone_centroids(
        self,
        zones_df: pd.DataFrame,
        connect_radius_km: float = 2.0,
    ) -> RoadNetwork:
        """Add zone centroids to network.

        Args:
            zones_df: Zones DataFrame with zone_id, lat, lon
            connect_radius_km: Radius for connecting centroids to network

        Returns:
            Updated RoadNetwork
        """
        from congestion_pricing.utils.geo import haversine_distance

        for _, zone in zones_df.iterrows():
            centroid_id = f"c_{zone['zone_id']}"

            # Add centroid node
            self.network.add_node(
                node_id=centroid_id,
                lat=zone["lat"],
                lon=zone["lon"],
                is_centroid=True,
                zone_id=zone["zone_id"],
            )

            # Find nearby nodes to connect
            for node_id, data in self.network.graph.nodes(data=True):
                if data.get("is_centroid", False):
                    continue

                dist = haversine_distance(
                    zone["lat"], zone["lon"],
                    data.get("lat", 0), data.get("lon", 0),
                )

                if dist <= connect_radius_km:
                    # Add connector edges
                    fft = dist / 30 * 60  # Assume 30 km/h on connectors

                    self.network.add_edge(
                        from_node=centroid_id,
                        to_node=node_id,
                        length_m=dist * 1000,
                        capacity_vph=10000,  # High capacity for connectors
                        fft_minutes=fft,
                        road_class="connector",
                    )
                    self.network.add_edge(
                        from_node=node_id,
                        to_node=centroid_id,
                        length_m=dist * 1000,
                        capacity_vph=10000,
                        fft_minutes=fft,
                        road_class="connector",
                    )

        return self.network
