"""Data schemas for the congestion pricing framework.

This module defines the standardized schemas for all data products
using Pydantic for validation.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class EdgeSchema(BaseModel):
    """Schema for network edge data.

    Attributes:
        edge_id: Unique edge identifier
        from_node: Origin node ID
        to_node: Destination node ID
        length_m: Edge length in meters
        lanes: Number of lanes
        speed_limit_kmh: Posted speed limit in km/h
        road_class: Road classification (e.g., 'primary', 'secondary')
        capacity_vph: Estimated capacity in vehicles per hour
        fft_minutes: Free-flow travel time in minutes
        in_toll_zone: Whether edge is inside congestion charge zone
    """

    edge_id: str
    from_node: str
    to_node: str
    length_m: float = Field(ge=0)
    lanes: int = Field(ge=1, default=1)
    speed_limit_kmh: float = Field(ge=0, default=50.0)
    road_class: str = "unclassified"
    capacity_vph: float = Field(ge=0, default=1800.0)
    fft_minutes: float = Field(ge=0)
    in_toll_zone: bool = False

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> list[str]:
        """Validate a DataFrame against the schema.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation error messages
        """
        errors = []
        required_cols = ["edge_id", "from_node", "to_node", "length_m"]

        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if "length_m" in df.columns and (df["length_m"] < 0).any():
            errors.append("Negative values in length_m")

        if "lanes" in df.columns and (df["lanes"] < 1).any():
            errors.append("lanes must be >= 1")

        return errors


class NodeSchema(BaseModel):
    """Schema for network node data.

    Attributes:
        node_id: Unique node identifier
        lat: Latitude
        lon: Longitude
        is_centroid: Whether this is a zone centroid
        zone_id: Zone ID if this is a centroid
    """

    node_id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    is_centroid: bool = False
    zone_id: str | None = None

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> list[str]:
        """Validate a DataFrame against the schema."""
        errors = []
        required_cols = ["node_id", "lat", "lon"]

        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if "lat" in df.columns:
            if (df["lat"] < -90).any() or (df["lat"] > 90).any():
                errors.append("lat must be between -90 and 90")

        if "lon" in df.columns:
            if (df["lon"] < -180).any() or (df["lon"] > 180).any():
                errors.append("lon must be between -180 and 180")

        return errors


class ODMatrixSchema(BaseModel):
    """Schema for OD demand matrix.

    Attributes:
        origin_zone: Origin zone ID
        dest_zone: Destination zone ID
        period: Time period identifier
        demand_auto: Auto person-trips
        demand_transit: Transit trips
        demand_other: Walk/bike/other trips
    """

    origin_zone: str
    dest_zone: str
    period: str
    demand_auto: float = Field(ge=0, default=0.0)
    demand_transit: float = Field(ge=0, default=0.0)
    demand_other: float = Field(ge=0, default=0.0)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> list[str]:
        """Validate a DataFrame against the schema."""
        errors = []
        required_cols = ["origin_zone", "dest_zone", "period"]

        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        demand_cols = ["demand_auto", "demand_transit", "demand_other"]
        for col in demand_cols:
            if col in df.columns and (df[col] < 0).any():
                errors.append(f"Negative values in {col}")

        return errors


class ZoneSchema(BaseModel):
    """Schema for zone definitions.

    Attributes:
        zone_id: Unique zone identifier
        name: Zone name
        population: Resident population
        employment: Jobs in zone
        median_income: Median household income
        pct_no_vehicle: Percentage of households without car
        income_quintile: Income quintile (1-5)
        area_sqkm: Zone area in square kilometers
    """

    zone_id: str
    name: str = ""
    population: int = Field(ge=0, default=0)
    employment: int = Field(ge=0, default=0)
    median_income: float = Field(ge=0, default=0.0)
    pct_no_vehicle: float = Field(ge=0, le=1, default=0.0)
    income_quintile: int = Field(ge=1, le=5, default=3)
    area_sqkm: float = Field(ge=0, default=0.0)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> list[str]:
        """Validate a DataFrame against the schema."""
        errors = []

        if "zone_id" not in df.columns:
            errors.append("Missing required column: zone_id")

        if "pct_no_vehicle" in df.columns:
            if (df["pct_no_vehicle"] < 0).any() or (df["pct_no_vehicle"] > 1).any():
                errors.append("pct_no_vehicle must be between 0 and 1")

        if "income_quintile" in df.columns:
            if (df["income_quintile"] < 1).any() or (df["income_quintile"] > 5).any():
                errors.append("income_quintile must be between 1 and 5")

        return errors


class TrafficObservationSchema(BaseModel):
    """Schema for traffic observations.

    Attributes:
        location_id: Observation location identifier
        edge_id: Matched edge ID (if mapped)
        timestamp: Observation timestamp
        speed_kmh: Observed speed in km/h
        count: Vehicle count
        sample_size: Number of samples for speed observation
    """

    location_id: str
    edge_id: str | None = None
    timestamp: str  # ISO format
    speed_kmh: float | None = Field(ge=0, default=None)
    count: int | None = Field(ge=0, default=None)
    sample_size: int = Field(ge=1, default=1)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> list[str]:
        """Validate a DataFrame against the schema."""
        errors = []
        required_cols = ["location_id", "timestamp"]

        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if "speed_kmh" in df.columns and (df["speed_kmh"].dropna() < 0).any():
            errors.append("Negative values in speed_kmh")

        if "count" in df.columns and (df["count"].dropna() < 0).any():
            errors.append("Negative values in count")

        return errors


def validate_dataframe(df: pd.DataFrame, schema_class: type[BaseModel]) -> list[str]:
    """Generic DataFrame validation against a schema.

    Args:
        df: DataFrame to validate
        schema_class: Schema class with validate_dataframe method

    Returns:
        List of validation errors
    """
    if hasattr(schema_class, "validate_dataframe"):
        return schema_class.validate_dataframe(df)
    return []


def dataframe_to_records(df: pd.DataFrame, schema_class: type[BaseModel]) -> list[BaseModel]:
    """Convert DataFrame to list of schema objects.

    Args:
        df: DataFrame to convert
        schema_class: Schema class to instantiate

    Returns:
        List of validated schema objects
    """
    records = []
    for _, row in df.iterrows():
        records.append(schema_class(**row.to_dict()))
    return records
