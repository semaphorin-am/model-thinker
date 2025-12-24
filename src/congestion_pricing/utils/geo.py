"""Geospatial utilities."""

from typing import Any

import numpy as np


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    earth_radius_km: float = 6371.0,
) -> float:
    """Calculate haversine distance between two points.

    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        earth_radius_km: Earth radius in km

    Returns:
        Distance in km
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    """Check if a point is inside a polygon using ray casting.

    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def bounding_box(
    coords: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Calculate bounding box for a set of coordinates.

    Args:
        coords: List of (lon, lat) coordinates

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def load_geojson(path: str) -> dict[str, Any]:
    """Load a GeoJSON file.

    Args:
        path: Path to GeoJSON file

    Returns:
        Parsed GeoJSON as dictionary
    """
    import json

    with open(path) as f:
        return json.load(f)


def polygon_from_geojson(geojson: dict[str, Any]) -> list[tuple[float, float]]:
    """Extract polygon coordinates from GeoJSON.

    Args:
        geojson: GeoJSON dictionary

    Returns:
        List of (lon, lat) coordinates

    Raises:
        ValueError: If geometry type is not Polygon
    """
    if geojson.get("type") == "Feature":
        geometry = geojson["geometry"]
    elif geojson.get("type") == "FeatureCollection":
        geometry = geojson["features"][0]["geometry"]
    else:
        geometry = geojson

    if geometry["type"] != "Polygon":
        raise ValueError(f"Expected Polygon, got {geometry['type']}")

    # Return exterior ring
    return [(coord[0], coord[1]) for coord in geometry["coordinates"][0]]
