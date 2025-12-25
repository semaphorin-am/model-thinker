"""Core plotting functions for congestion pricing results.

This module provides reusable plotting components using Plotly
for interactive visualization.
"""

from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from congestion_pricing.policy.results import ModelResult


def _check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly"
        )


def plot_mode_shares(
    results: ModelResult | list[ModelResult],
    title: str = "Mode Share Distribution",
) -> "go.Figure":
    """Plot mode share as a stacked bar or grouped bar chart.

    Args:
        results: Single result or list of results to compare
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    if isinstance(results, ModelResult):
        results = [results]

    modes = ["Auto", "Transit", "Walk/Cycle", "Other"]

    fig = go.Figure()

    for result in results:
        agg = result.aggregates
        shares = [
            agg.mode_share_auto,
            agg.mode_share_transit,
            getattr(agg, "mode_share_walk_cycle", 0.0),
            1 - agg.mode_share_auto - agg.mode_share_transit - getattr(agg, "mode_share_walk_cycle", 0.0),
        ]
        # Ensure non-negative
        shares = [max(0, s) for s in shares]

        fig.add_trace(go.Bar(
            name=f"{result.model_name} - {result.scenario_name}",
            x=modes,
            y=shares,
            text=[f"{s:.1%}" for s in shares],
            textposition="auto",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Mode",
        yaxis_title="Share",
        yaxis_tickformat=".0%",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def plot_travel_time_distribution(
    results: ModelResult,
    title: str = "Travel Time Distribution",
) -> "go.Figure":
    """Plot travel time distribution histogram.

    Args:
        results: Model results with travel time data
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    fig = go.Figure()

    # If we have trajectory data with travel times
    if results.trajectory and "travel_times" in results.trajectory[0]:
        all_times = []
        for day_data in results.trajectory:
            all_times.extend(day_data.get("travel_times", []))

        fig.add_trace(go.Histogram(
            x=all_times,
            nbinsx=30,
            name="Travel Times",
            marker_color="steelblue",
        ))

        mean_time = np.mean(all_times)
        fig.add_vline(
            x=mean_time,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_time:.1f} min",
        )
    else:
        # Use aggregate statistics
        mean_tt = results.aggregates.mean_travel_time
        std_tt = getattr(results.aggregates, "std_travel_time", mean_tt * 0.2)

        x = np.linspace(max(0, mean_tt - 3*std_tt), mean_tt + 3*std_tt, 100)
        y = np.exp(-0.5 * ((x - mean_tt) / std_tt) ** 2) / (std_tt * np.sqrt(2 * np.pi))

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="tozeroy",
            name="Estimated Distribution",
            line_color="steelblue",
        ))

        fig.add_vline(
            x=mean_tt,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_tt:.1f} min",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Travel Time (minutes)",
        yaxis_title="Frequency / Density",
    )

    return fig


def plot_toll_revenue(
    results: list[ModelResult],
    title: str = "Toll Revenue Comparison",
) -> "go.Figure":
    """Plot toll revenue across scenarios/models.

    Args:
        results: List of model results
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    names = [f"{r.model_name}\n{r.scenario_name}" for r in results]
    revenues = [r.aggregates.toll_revenue for r in results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=revenues,
        marker_color="forestgreen",
        text=[f"${r:,.0f}" for r in revenues],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Model / Scenario",
        yaxis_title="Daily Toll Revenue ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    )

    return fig


def plot_link_flows(
    results: ModelResult,
    top_n: int = 20,
    title: str = "Link Flow Comparison",
) -> "go.Figure":
    """Plot link flows for top congested links.

    Args:
        results: Model results with link-level data
        top_n: Number of top links to show
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    if results.link_results is None:
        raise ValueError("Results do not contain link-level data")

    flows = results.link_results.flows
    capacities = getattr(results.link_results, "capacities", {})

    # Sort by flow
    sorted_links = sorted(flows.items(), key=lambda x: x[1], reverse=True)[:top_n]

    link_names = [str(link) for link, _ in sorted_links]
    flow_values = [flow for _, flow in sorted_links]
    cap_values = [capacities.get(link, flow * 1.2) for link, flow in sorted_links]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Flow",
        x=link_names,
        y=flow_values,
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="Capacity",
        x=link_names,
        y=cap_values,
        marker_color="lightgray",
        opacity=0.5,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Link",
        yaxis_title="Flow (veh/hr)",
        barmode="overlay",
        xaxis_tickangle=-45,
    )

    return fig


def plot_convergence(
    results: ModelResult,
    title: str = "Model Convergence",
) -> "go.Figure":
    """Plot convergence history for iterative models.

    Args:
        results: Model results with convergence data
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    fig = go.Figure()

    # Check for convergence data in parameters
    conv_history = results.parameters.get("convergence_history", [])

    if conv_history:
        iterations = list(range(1, len(conv_history) + 1))

        fig.add_trace(go.Scatter(
            x=iterations,
            y=conv_history,
            mode="lines+markers",
            name="Gap",
            line=dict(color="steelblue"),
        ))

        # Add convergence threshold line
        threshold = results.parameters.get("convergence_threshold", 0.001)
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}",
        )

    elif results.trajectory:
        # Use trajectory data if available
        metrics = []
        for day_data in results.trajectory:
            if "gap" in day_data:
                metrics.append(day_data["gap"])
            elif "mode_share_auto" in day_data:
                metrics.append(day_data["mode_share_auto"])

        if metrics:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(metrics) + 1)),
                y=metrics,
                mode="lines+markers",
                name="Metric",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Convergence Gap",
        yaxis_type="log",
    )

    return fig


def plot_emissions(
    results: list[ModelResult],
    title: str = "Emissions Comparison",
) -> "go.Figure":
    """Plot emissions across scenarios.

    Args:
        results: List of model results
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    names = [f"{r.model_name}\n{r.scenario_name}" for r in results]
    co2 = [getattr(r.aggregates, "emissions_co2_kg", 0) for r in results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=co2,
        marker_color="darkgreen",
        text=[f"{e:,.0f} kg" for e in co2],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Model / Scenario",
        yaxis_title="CO2 Emissions (kg)",
    )

    return fig


def plot_equity_metrics(
    results: ModelResult,
    title: str = "Equity Analysis",
) -> "go.Figure":
    """Plot equity metrics by income group.

    Args:
        results: Model results with equity data
        title: Plot title

    Returns:
        Plotly figure
    """
    _check_plotly()

    equity = results.parameters.get("equity_metrics", {})

    if not equity:
        # Generate placeholder
        equity = {
            "Low Income": {"burden": 0.08, "benefit": 0.02},
            "Middle Income": {"burden": 0.05, "benefit": 0.04},
            "High Income": {"burden": 0.03, "benefit": 0.06},
        }

    groups = list(equity.keys())
    burdens = [equity[g].get("burden", 0) for g in groups]
    benefits = [equity[g].get("benefit", 0) for g in groups]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Toll Burden (% income)",
        x=groups,
        y=burdens,
        marker_color="indianred",
    ))

    fig.add_trace(go.Bar(
        name="Time Savings Benefit",
        x=groups,
        y=benefits,
        marker_color="seagreen",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Income Group",
        yaxis_title="Percentage",
        yaxis_tickformat=".1%",
        barmode="group",
    )

    return fig


def create_summary_card(
    results: ModelResult,
) -> "go.Figure":
    """Create a summary metrics card.

    Args:
        results: Model results

    Returns:
        Plotly figure with indicator cards
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "indicator"}] * 3] * 2,
        subplot_titles=[
            "Mean Travel Time",
            "Auto Mode Share",
            "Transit Mode Share",
            "Toll Revenue",
            "Convergence",
            "Runtime",
        ],
    )

    agg = results.aggregates

    # Travel time
    fig.add_trace(go.Indicator(
        mode="number",
        value=agg.mean_travel_time,
        number={"suffix": " min"},
    ), row=1, col=1)

    # Auto share
    fig.add_trace(go.Indicator(
        mode="number",
        value=agg.mode_share_auto * 100,
        number={"suffix": "%"},
    ), row=1, col=2)

    # Transit share
    fig.add_trace(go.Indicator(
        mode="number",
        value=agg.mode_share_transit * 100,
        number={"suffix": "%"},
    ), row=1, col=3)

    # Revenue
    fig.add_trace(go.Indicator(
        mode="number",
        value=agg.toll_revenue,
        number={"prefix": "$", "valueformat": ",.0f"},
    ), row=2, col=1)

    # Convergence
    fig.add_trace(go.Indicator(
        mode="number",
        value=1 if agg.converged else 0,
        number={"valueformat": ".0f"},
    ), row=2, col=2)

    # Runtime
    fig.add_trace(go.Indicator(
        mode="number",
        value=agg.runtime_seconds,
        number={"suffix": "s", "valueformat": ".2f"},
    ), row=2, col=3)

    fig.update_layout(
        title=f"Results Summary: {results.model_name} - {results.scenario_name}",
        height=400,
    )

    return fig
