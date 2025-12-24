"""Data processing utilities.

This module provides functionality for processing raw data into
standardized formats for use in the modeling framework.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.data.schemas import (
    EdgeSchema,
    NodeSchema,
    ODMatrixSchema,
    ZoneSchema,
)
from congestion_pricing.utils.io import ensure_dir, load_yaml
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


# Road class to capacity mapping (vehicles per hour per lane)
CAPACITY_BY_CLASS = {
    "motorway": 2000,
    "motorway_link": 1800,
    "trunk": 1800,
    "trunk_link": 1600,
    "primary": 900,
    "primary_link": 800,
    "secondary": 700,
    "secondary_link": 600,
    "tertiary": 500,
    "tertiary_link": 400,
    "residential": 300,
    "unclassified": 300,
    "service": 200,
}

# Default speed limits by road class (km/h)
SPEED_BY_CLASS = {
    "motorway": 112,  # 70 mph
    "motorway_link": 80,
    "trunk": 80,
    "trunk_link": 64,
    "primary": 64,  # 40 mph
    "primary_link": 48,
    "secondary": 48,  # 30 mph
    "secondary_link": 40,
    "tertiary": 40,
    "tertiary_link": 32,
    "residential": 32,  # 20 mph
    "unclassified": 32,
    "service": 24,
}


class DataProcessor:
    """Process raw data into model-ready formats."""

    def __init__(
        self,
        city: str,
        raw_dir: str | Path = "data/raw",
        output_dir: str | Path = "data/processed",
    ):
        """Initialize processor.

        Args:
            city: City identifier
            raw_dir: Directory containing raw data
            output_dir: Directory for processed output
        """
        self.city = city.lower()
        self.raw_dir = Path(raw_dir) / self.city
        self.output_dir = Path(output_dir) / self.city

    def process_network_from_osm(
        self,
        osm_file: str | Path | None = None,
        network_type: str = "drive",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process OSM data into nodes and edges.

        Args:
            osm_file: Path to OSM file (default: auto-detect)
            network_type: OSMnx network type

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("osmnx is required for OSM processing")

        if osm_file is None:
            osm_file = self.raw_dir / f"{self.city}.osm.pbf"

        logger.info(f"Processing OSM network from {osm_file}")

        # Load network
        if Path(osm_file).exists():
            G = ox.graph_from_xml(str(osm_file), simplify=True)
        else:
            # Fall back to downloading by city name
            logger.warning(f"OSM file not found, downloading for {self.city}")
            G = ox.graph_from_place(self.city, network_type=network_type)

        # Project to local CRS for accurate distances
        G = ox.project_graph(G)

        # Convert to DataFrames
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

        # Process nodes
        nodes_df = pd.DataFrame({
            "node_id": nodes_gdf.index.astype(str),
            "lat": nodes_gdf.geometry.y if hasattr(nodes_gdf.geometry, "y") else nodes_gdf["y"],
            "lon": nodes_gdf.geometry.x if hasattr(nodes_gdf.geometry, "x") else nodes_gdf["x"],
            "is_centroid": False,
            "zone_id": None,
        })

        # Process edges
        edges_df = self._process_edges(edges_gdf)

        # Validate
        node_errors = NodeSchema.validate_dataframe(nodes_df)
        edge_errors = EdgeSchema.validate_dataframe(edges_df)

        if node_errors:
            logger.warning(f"Node validation warnings: {node_errors}")
        if edge_errors:
            logger.warning(f"Edge validation warnings: {edge_errors}")

        # Save
        output_dir = ensure_dir(self.output_dir / "network")
        nodes_df.to_parquet(output_dir / "nodes.parquet", index=False)
        edges_df.to_parquet(output_dir / "edges.parquet", index=False)

        logger.info(f"Processed {len(nodes_df)} nodes and {len(edges_df)} edges")

        return nodes_df, edges_df

    def _process_edges(self, edges_gdf: Any) -> pd.DataFrame:
        """Process edges GeoDataFrame into standardized format.

        Args:
            edges_gdf: GeoDataFrame from OSMnx

        Returns:
            Processed edges DataFrame
        """
        edges_df = pd.DataFrame()

        # Create edge IDs from u, v, key
        if edges_gdf.index.nlevels == 3:
            edges_df["edge_id"] = [
                f"e_{u}_{v}_{k}" for u, v, k in edges_gdf.index
            ]
            edges_df["from_node"] = [str(u) for u, v, k in edges_gdf.index]
            edges_df["to_node"] = [str(v) for u, v, k in edges_gdf.index]
        else:
            edges_df["edge_id"] = [f"e_{i}" for i in range(len(edges_gdf))]
            edges_df["from_node"] = edges_gdf["u"].astype(str).values
            edges_df["to_node"] = edges_gdf["v"].astype(str).values

        # Length
        edges_df["length_m"] = edges_gdf["length"].values

        # Lanes
        if "lanes" in edges_gdf.columns:
            edges_df["lanes"] = (
                edges_gdf["lanes"]
                .apply(lambda x: int(x[0]) if isinstance(x, list) else (int(x) if pd.notna(x) else 1))
                .values
            )
        else:
            edges_df["lanes"] = 1

        # Road class
        if "highway" in edges_gdf.columns:
            edges_df["road_class"] = (
                edges_gdf["highway"]
                .apply(lambda x: x[0] if isinstance(x, list) else x)
                .fillna("unclassified")
                .values
            )
        else:
            edges_df["road_class"] = "unclassified"

        # Speed limit
        if "maxspeed" in edges_gdf.columns:
            edges_df["speed_limit_kmh"] = (
                edges_gdf["maxspeed"]
                .apply(self._parse_speed)
                .values
            )
        else:
            edges_df["speed_limit_kmh"] = edges_df["road_class"].map(
                lambda x: SPEED_BY_CLASS.get(x, 32)
            )

        # Fill missing speeds from road class
        mask = edges_df["speed_limit_kmh"].isna()
        edges_df.loc[mask, "speed_limit_kmh"] = edges_df.loc[mask, "road_class"].map(
            lambda x: SPEED_BY_CLASS.get(x, 32)
        )

        # Capacity
        edges_df["capacity_vph"] = (
            edges_df["road_class"].map(lambda x: CAPACITY_BY_CLASS.get(x, 300))
            * edges_df["lanes"]
        )

        # Free-flow travel time (minutes)
        edges_df["fft_minutes"] = (
            edges_df["length_m"] / 1000 / edges_df["speed_limit_kmh"] * 60
        )

        # Toll zone flag (default false, set later)
        edges_df["in_toll_zone"] = False

        return edges_df

    def _parse_speed(self, value: Any) -> float | None:
        """Parse speed limit from OSM format.

        Args:
            value: Speed value (may be string like '30 mph')

        Returns:
            Speed in km/h or None
        """
        if pd.isna(value):
            return None

        if isinstance(value, list):
            value = value[0]

        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).lower().strip()

        try:
            if "mph" in value:
                return float(value.replace("mph", "").strip()) * 1.60934
            elif "km/h" in value or "kmh" in value:
                return float(value.replace("km/h", "").replace("kmh", "").strip())
            else:
                return float(value)
        except (ValueError, TypeError):
            return None

    def create_synthetic_od_matrix(
        self,
        zones_df: pd.DataFrame,
        gravity_beta: float = 0.1,
        total_trips: float = 100000.0,
    ) -> pd.DataFrame:
        """Create synthetic OD matrix using gravity model.

        Args:
            zones_df: DataFrame with zone attributes
            gravity_beta: Distance decay parameter
            total_trips: Total trips to generate

        Returns:
            OD matrix DataFrame
        """
        logger.info("Generating synthetic OD matrix")

        n_zones = len(zones_df)

        # Simple gravity model: Tij = k * Pi * Ej * exp(-beta * dij)
        # For simplicity, use population as production and employment as attraction

        populations = zones_df["population"].fillna(1000).values
        employments = zones_df["employment"].fillna(500).values

        # Simple distance proxy (zone index difference - replace with actual distances)
        distances = np.abs(np.arange(n_zones)[:, None] - np.arange(n_zones)[None, :]) + 0.1

        # Gravity model
        trips = (
            populations[:, None]
            * employments[None, :]
            * np.exp(-gravity_beta * distances)
        )

        # Normalize to total trips
        trips = trips / trips.sum() * total_trips

        # Convert to long format
        records = []
        zone_ids = zones_df["zone_id"].values

        for period in ["am_peak", "pm_peak"]:
            period_factor = 0.3 if period == "am_peak" else 0.25
            for i, origin in enumerate(zone_ids):
                for j, dest in enumerate(zone_ids):
                    if trips[i, j] > 0.1:  # Skip very small flows
                        records.append({
                            "origin_zone": origin,
                            "dest_zone": dest,
                            "period": period,
                            "demand_auto": trips[i, j] * period_factor * 0.6,
                            "demand_transit": trips[i, j] * period_factor * 0.3,
                            "demand_other": trips[i, j] * period_factor * 0.1,
                        })

        od_df = pd.DataFrame(records)

        # Validate
        errors = ODMatrixSchema.validate_dataframe(od_df)
        if errors:
            logger.warning(f"OD matrix validation warnings: {errors}")

        # Save
        output_dir = ensure_dir(self.output_dir / "demand")
        od_df.to_parquet(output_dir / "od_matrix.parquet", index=False)

        logger.info(f"Generated OD matrix with {len(od_df)} OD pairs")

        return od_df

    def create_synthetic_zones(
        self,
        n_zones: int = 50,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> pd.DataFrame:
        """Create synthetic zone definitions.

        Args:
            n_zones: Number of zones to create
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Zones DataFrame
        """
        logger.info(f"Creating {n_zones} synthetic zones")

        if bbox is None:
            # Default to approximate city bounds
            if self.city == "london":
                bbox = (-0.5, 51.3, 0.3, 51.7)
            elif self.city == "nyc":
                bbox = (-74.3, 40.5, -73.7, 40.9)
            else:
                bbox = (-0.2, 51.4, 0.1, 51.6)

        np.random.seed(42)

        zones_df = pd.DataFrame({
            "zone_id": [f"zone_{i}" for i in range(n_zones)],
            "name": [f"Zone {i}" for i in range(n_zones)],
            "lon": np.random.uniform(bbox[0], bbox[2], n_zones),
            "lat": np.random.uniform(bbox[1], bbox[3], n_zones),
            "population": np.random.lognormal(8, 1, n_zones).astype(int),
            "employment": np.random.lognormal(7, 1.5, n_zones).astype(int),
            "median_income": np.random.lognormal(10.5, 0.5, n_zones),
            "pct_no_vehicle": np.random.beta(2, 5, n_zones),
            "area_sqkm": np.random.uniform(0.5, 5, n_zones),
        })

        # Assign income quintiles
        zones_df["income_quintile"] = pd.qcut(
            zones_df["median_income"], 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Validate
        errors = ZoneSchema.validate_dataframe(zones_df)
        if errors:
            logger.warning(f"Zone validation warnings: {errors}")

        # Save
        output_dir = ensure_dir(self.output_dir / "zones")
        zones_df.to_parquet(output_dir / "zones.parquet", index=False)

        logger.info(f"Created {len(zones_df)} zones")

        return zones_df

    def process_all(self) -> dict[str, pd.DataFrame]:
        """Process all data for the city.

        Returns:
            Dictionary of processed DataFrames
        """
        results = {}

        # Try to process network
        try:
            nodes_df, edges_df = self.process_network_from_osm()
            results["nodes"] = nodes_df
            results["edges"] = edges_df
        except Exception as e:
            logger.warning(f"Could not process OSM network: {e}")
            logger.info("Creating synthetic data instead")

        # Create zones
        zones_df = self.create_synthetic_zones()
        results["zones"] = zones_df

        # Create OD matrix
        od_df = self.create_synthetic_od_matrix(zones_df)
        results["od_matrix"] = od_df

        return results


def process_city_data(
    city: str,
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    """Convenience function to process data for a city.

    Args:
        city: City identifier
        raw_dir: Raw data directory
        output_dir: Output directory

    Returns:
        Dictionary of processed DataFrames
    """
    processor = DataProcessor(city, raw_dir, output_dir)
    return processor.process_all()
