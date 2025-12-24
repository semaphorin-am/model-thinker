"""Zone system management.

This module provides utilities for working with traffic analysis zones,
including zone definitions, aggregation, and spatial operations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


class ZoneSystem:
    """Traffic Analysis Zone (TAZ) system.

    Manages zone definitions, attributes, and aggregation operations.
    """

    def __init__(self, zones_df: pd.DataFrame):
        """Initialize zone system.

        Args:
            zones_df: DataFrame with zone attributes
        """
        self.zones = zones_df.copy()
        self._index_by_id()

    def _index_by_id(self) -> None:
        """Create zone ID index."""
        if "zone_id" in self.zones.columns:
            self.zones = self.zones.set_index("zone_id", drop=False)

    @property
    def zone_ids(self) -> list[str]:
        """List of zone IDs."""
        return self.zones["zone_id"].tolist()

    @property
    def n_zones(self) -> int:
        """Number of zones."""
        return len(self.zones)

    def get_zone(self, zone_id: str) -> pd.Series:
        """Get attributes for a zone.

        Args:
            zone_id: Zone identifier

        Returns:
            Series of zone attributes
        """
        return self.zones.loc[zone_id]

    def get_income_quintile(self, zone_id: str) -> int:
        """Get income quintile for a zone.

        Args:
            zone_id: Zone identifier

        Returns:
            Income quintile (1-5)
        """
        return int(self.zones.loc[zone_id].get("income_quintile", 3))

    def zones_by_quintile(self, quintile: int) -> list[str]:
        """Get zones in a specific income quintile.

        Args:
            quintile: Income quintile (1-5)

        Returns:
            List of zone IDs
        """
        mask = self.zones["income_quintile"] == quintile
        return self.zones[mask]["zone_id"].tolist()

    def aggregate_by_quintile(
        self,
        values: pd.Series,
        agg_func: str = "sum",
    ) -> pd.Series:
        """Aggregate zone values by income quintile.

        Args:
            values: Series indexed by zone_id
            agg_func: Aggregation function ('sum', 'mean', 'count')

        Returns:
            Series indexed by quintile
        """
        # Join values with zone quintiles
        df = pd.DataFrame({
            "value": values,
            "quintile": self.zones.loc[values.index, "income_quintile"],
        })

        return df.groupby("quintile")["value"].agg(agg_func)

    def get_centroids(self) -> pd.DataFrame:
        """Get zone centroids.

        Returns:
            DataFrame with zone_id, lat, lon
        """
        return self.zones[["zone_id", "lat", "lon"]].copy()

    def compute_distance_matrix(self) -> pd.DataFrame:
        """Compute zone-to-zone distance matrix.

        Returns:
            DataFrame with origin, destination, distance_km
        """
        from congestion_pricing.utils.geo import haversine_distance

        records = []
        zone_ids = self.zone_ids
        coords = self.zones[["lat", "lon"]].values

        for i, origin in enumerate(zone_ids):
            for j, dest in enumerate(zone_ids):
                dist = haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1],
                )
                records.append({
                    "origin": origin,
                    "destination": dest,
                    "distance_km": dist,
                })

        return pd.DataFrame(records)

    def get_population(self, zone_id: str | None = None) -> float | pd.Series:
        """Get population for zone(s).

        Args:
            zone_id: Zone identifier, or None for all zones

        Returns:
            Population value or Series
        """
        if zone_id is not None:
            return self.zones.loc[zone_id, "population"]
        return self.zones["population"]

    def get_employment(self, zone_id: str | None = None) -> float | pd.Series:
        """Get employment for zone(s).

        Args:
            zone_id: Zone identifier, or None for all zones

        Returns:
            Employment value or Series
        """
        if zone_id is not None:
            return self.zones.loc[zone_id, "employment"]
        return self.zones["employment"]

    def total_population(self) -> float:
        """Get total population across all zones."""
        return self.zones["population"].sum()

    def total_employment(self) -> float:
        """Get total employment across all zones."""
        return self.zones["employment"].sum()

    def save(self, path: str | Path) -> None:
        """Save zone system to parquet.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.zones.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "ZoneSystem":
        """Load zone system from parquet.

        Args:
            path: Path to parquet file

        Returns:
            Loaded ZoneSystem
        """
        zones_df = pd.read_parquet(path)
        return cls(zones_df)

    @classmethod
    def from_geojson(cls, path: str | Path) -> "ZoneSystem":
        """Load zone system from GeoJSON.

        Args:
            path: Path to GeoJSON file

        Returns:
            Loaded ZoneSystem
        """
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path)

            # Compute centroids if geometry present
            if "geometry" in gdf.columns:
                gdf["lat"] = gdf.geometry.centroid.y
                gdf["lon"] = gdf.geometry.centroid.x

            return cls(pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore")))

        except ImportError:
            raise ImportError("geopandas is required for GeoJSON loading")


def create_grid_zones(
    bbox: tuple[float, float, float, float],
    n_x: int = 10,
    n_y: int = 10,
) -> ZoneSystem:
    """Create a grid of zones covering a bounding box.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        n_x: Number of zones in x direction
        n_y: Number of zones in y direction

    Returns:
        ZoneSystem with grid zones
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    lon_step = (max_lon - min_lon) / n_x
    lat_step = (max_lat - min_lat) / n_y

    records = []
    idx = 0

    for i in range(n_x):
        for j in range(n_y):
            lon = min_lon + (i + 0.5) * lon_step
            lat = min_lat + (j + 0.5) * lat_step

            records.append({
                "zone_id": f"zone_{idx}",
                "name": f"Grid Zone ({i}, {j})",
                "lat": lat,
                "lon": lon,
                "population": 1000,
                "employment": 500,
                "median_income": 50000,
                "pct_no_vehicle": 0.2,
                "income_quintile": (idx % 5) + 1,
                "area_sqkm": 111.0 * lat_step * 111.0 * lon_step * np.cos(np.radians(lat)),
            })
            idx += 1

    return ZoneSystem(pd.DataFrame(records))
