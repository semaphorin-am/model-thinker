"""Capacity estimation and travel time functions.

This module provides utilities for estimating road capacity and
computing congestion-dependent travel times.
"""

import numpy as np


def estimate_capacity(
    lanes: int,
    road_class: str,
    base_capacity_per_lane: dict[str, float] | None = None,
) -> float:
    """Estimate link capacity based on lanes and road class.

    Args:
        lanes: Number of lanes
        road_class: Road classification
        base_capacity_per_lane: Optional custom capacity values

    Returns:
        Capacity in vehicles per hour
    """
    if base_capacity_per_lane is None:
        base_capacity_per_lane = {
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
            "connector": 10000,  # High capacity for zone connectors
        }

    per_lane = base_capacity_per_lane.get(road_class, 300)
    return lanes * per_lane


def bpr_travel_time(
    free_flow_time: float | np.ndarray,
    flow: float | np.ndarray,
    capacity: float | np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float | np.ndarray:
    """Compute travel time using Bureau of Public Roads (BPR) function.

    t = t0 * (1 + alpha * (v/c)^beta)

    Args:
        free_flow_time: Free-flow travel time
        flow: Current flow
        capacity: Link capacity
        alpha: BPR alpha parameter (default 0.15)
        beta: BPR beta parameter (default 4.0)

    Returns:
        Travel time
    """
    # Handle division by zero
    capacity = np.maximum(capacity, 1e-6)
    vc_ratio = np.clip(flow / capacity, 0, 10)  # Cap at 10 to avoid numerical issues

    return free_flow_time * (1 + alpha * np.power(vc_ratio, beta))


def bpr_derivative(
    free_flow_time: float | np.ndarray,
    flow: float | np.ndarray,
    capacity: float | np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float | np.ndarray:
    """Compute derivative of BPR function with respect to flow.

    dt/dv = t0 * alpha * beta * (v/c)^(beta-1) / c

    Args:
        free_flow_time: Free-flow travel time
        flow: Current flow
        capacity: Link capacity
        alpha: BPR alpha parameter
        beta: BPR beta parameter

    Returns:
        Derivative of travel time with respect to flow
    """
    capacity = np.maximum(capacity, 1e-6)
    vc_ratio = np.clip(flow / capacity, 0, 10)

    return free_flow_time * alpha * beta * np.power(vc_ratio, beta - 1) / capacity


def bpr_integral(
    free_flow_time: float | np.ndarray,
    flow: float | np.ndarray,
    capacity: float | np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float | np.ndarray:
    """Compute integral of BPR function (for Beckmann objective).

    integral_0^v t(w) dw = t0 * v * (1 + alpha/(beta+1) * (v/c)^beta)

    Args:
        free_flow_time: Free-flow travel time
        flow: Current flow
        capacity: Link capacity
        alpha: BPR alpha parameter
        beta: BPR beta parameter

    Returns:
        Integral value
    """
    capacity = np.maximum(capacity, 1e-6)
    vc_ratio = np.clip(flow / capacity, 0, 10)

    return free_flow_time * flow * (1 + alpha / (beta + 1) * np.power(vc_ratio, beta))


def generalized_cost(
    travel_time: float | np.ndarray,
    toll: float | np.ndarray,
    vot: float = 20.0,
    distance_cost: float = 0.0,
    distance: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute generalized cost of travel.

    GC = time + toll/VOT + distance_cost * distance / VOT

    Args:
        travel_time: Travel time in minutes
        toll: Toll in currency units
        vot: Value of time ($/hour)
        distance_cost: Per-km operating cost ($)
        distance: Distance in km

    Returns:
        Generalized cost in minutes-equivalent
    """
    vot_per_min = vot / 60
    return travel_time + toll / vot_per_min + distance_cost * distance / vot_per_min


def speed_from_flow(
    flow: float | np.ndarray,
    capacity: float | np.ndarray,
    free_flow_speed: float | np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float | np.ndarray:
    """Compute speed from flow using BPR relationship.

    Args:
        flow: Current flow
        capacity: Link capacity
        free_flow_speed: Free-flow speed (km/h)
        alpha: BPR alpha parameter
        beta: BPR beta parameter

    Returns:
        Speed in km/h
    """
    capacity = np.maximum(capacity, 1e-6)
    vc_ratio = np.clip(flow / capacity, 0, 10)

    return free_flow_speed / (1 + alpha * np.power(vc_ratio, beta))


def emission_rate(
    speed_kmh: float | np.ndarray,
    emission_type: str = "co2",
) -> float | np.ndarray:
    """Estimate emission rate based on average speed.

    Uses simplified speed-emission curve.

    Args:
        speed_kmh: Average speed in km/h
        emission_type: Type of emission ('co2', 'nox', 'pm25')

    Returns:
        Emission rate in g/km
    """
    # Simplified emission curves (actual values vary by vehicle type)
    speed = np.maximum(speed_kmh, 5)  # Minimum speed for calculation

    if emission_type == "co2":
        # U-shaped curve with minimum around 60 km/h
        return 100 + 500 / speed + 0.01 * speed ** 1.5
    elif emission_type == "nox":
        return 0.1 + 5 / speed + 0.001 * speed ** 1.3
    elif emission_type == "pm25":
        return 0.01 + 0.5 / speed + 0.0001 * speed ** 1.2
    else:
        raise ValueError(f"Unknown emission type: {emission_type}")


def total_emissions(
    flow: np.ndarray,
    distance: np.ndarray,
    speed: np.ndarray,
    emission_type: str = "co2",
) -> float:
    """Compute total emissions for a network.

    Args:
        flow: Link flows (vehicles/hour)
        distance: Link distances (km)
        speed: Link speeds (km/h)
        emission_type: Type of emission

    Returns:
        Total emissions in kg/hour
    """
    rate = emission_rate(speed, emission_type)  # g/km
    vkt = flow * distance  # vehicle-km/hour
    return np.sum(rate * vkt) / 1000  # kg/hour
