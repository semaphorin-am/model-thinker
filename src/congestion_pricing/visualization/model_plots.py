"""Model-specific visualization functions.

This module provides specialized plots for each model family,
highlighting their unique outputs and characteristics.
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
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required for visualization")


def plot_demand_response(
    results: ModelResult,
    toll_range: tuple[float, float] = (0, 25),
) -> "go.Figure":
    """Plot demand response curves showing elasticity.

    Args:
        results: Results from demand response model
        toll_range: Range of tolls to plot

    Returns:
        Plotly figure with demand curves
    """
    _check_plotly()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Demand vs Toll", "Mode Share vs Toll"],
    )

    # Extract elasticity from parameters
    elasticity = results.parameters.get("elasticity_auto", -0.3)
    base_demand = results.parameters.get("base_demand", 100000)

    tolls = np.linspace(toll_range[0], toll_range[1], 50)
    # Demand = base * (1 + elasticity * toll / base_toll)
    demand = base_demand * np.exp(elasticity * tolls / 10)

    fig.add_trace(go.Scatter(
        x=tolls,
        y=demand,
        mode="lines",
        name="Auto Demand",
        line=dict(color="steelblue", width=2),
    ), row=1, col=1)

    # Add current point
    current_toll = results.parameters.get("current_toll", 15)
    current_demand = base_demand * np.exp(elasticity * current_toll / 10)
    fig.add_trace(go.Scatter(
        x=[current_toll],
        y=[current_demand],
        mode="markers",
        name="Current",
        marker=dict(size=12, color="red"),
    ), row=1, col=1)

    # Mode share curve
    base_auto = 0.5
    auto_share = base_auto * np.exp(elasticity * tolls / 10)
    transit_share = 1 - auto_share

    fig.add_trace(go.Scatter(
        x=tolls,
        y=auto_share,
        mode="lines",
        name="Auto Share",
        line=dict(color="indianred"),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=tolls,
        y=transit_share,
        mode="lines",
        name="Transit Share",
        line=dict(color="seagreen"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Toll ($)", row=1, col=1)
    fig.update_xaxes(title_text="Toll ($)", row=1, col=2)
    fig.update_yaxes(title_text="Daily Trips", row=1, col=1)
    fig.update_yaxes(title_text="Mode Share", tickformat=".0%", row=1, col=2)

    fig.update_layout(
        title="Demand Response Model Results",
        height=400,
        showlegend=True,
    )

    return fig


def plot_equilibrium(
    results: ModelResult,
) -> "go.Figure":
    """Plot equilibrium model results including convergence and flows.

    Args:
        results: Results from equilibrium model

    Returns:
        Plotly figure with equilibrium diagnostics
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Convergence History",
            "Volume/Capacity Distribution",
            "Travel Time vs Flow",
            "Link Flow Distribution",
        ],
    )

    # Convergence
    conv_history = results.parameters.get("convergence_history", [])
    if conv_history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(conv_history) + 1)),
            y=conv_history,
            mode="lines+markers",
            name="Gap",
            line=dict(color="steelblue"),
        ), row=1, col=1)

        threshold = results.parameters.get("convergence_threshold", 0.001)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=1, col=1)

    # V/C distribution
    if results.link_results:
        flows = list(results.link_results.flows.values())
        capacities = list(getattr(results.link_results, "capacities", {}).values())

        if capacities and len(capacities) == len(flows):
            vc_ratios = [f/c if c > 0 else 0 for f, c in zip(flows, capacities)]
        else:
            vc_ratios = [f / 1000 for f in flows]  # Assume capacity 1000

        fig.add_trace(go.Histogram(
            x=vc_ratios,
            nbinsx=20,
            name="V/C Ratio",
            marker_color="orange",
        ), row=1, col=2)

        # Travel time vs flow
        travel_times = list(getattr(results.link_results, "travel_times", {}).values())
        if travel_times and len(travel_times) == len(flows):
            fig.add_trace(go.Scatter(
                x=flows,
                y=travel_times,
                mode="markers",
                name="Links",
                marker=dict(size=4, color="steelblue", opacity=0.5),
            ), row=2, col=1)

        # Flow distribution
        fig.add_trace(go.Histogram(
            x=flows,
            nbinsx=30,
            name="Link Flows",
            marker_color="seagreen",
        ), row=2, col=2)

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Gap", type="log", row=1, col=1)
    fig.update_xaxes(title_text="V/C Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Flow (veh/hr)", row=2, col=1)
    fig.update_yaxes(title_text="Travel Time (min)", row=2, col=1)
    fig.update_xaxes(title_text="Flow (veh/hr)", row=2, col=2)

    fig.update_layout(
        title=f"Equilibrium Model: {results.scenario_name}",
        height=600,
        showlegend=False,
    )

    return fig


def plot_abm_trajectory(
    results: ModelResult,
) -> "go.Figure":
    """Plot agent-based model trajectory over time.

    Args:
        results: Results from ABM

    Returns:
        Plotly figure with ABM dynamics
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Mode Share Evolution",
            "Travel Time Evolution",
            "Agent State Distribution",
            "Learning Curves",
        ],
    )

    trajectory = results.trajectory or []

    if trajectory:
        days = list(range(1, len(trajectory) + 1))

        # Mode shares over time
        auto_shares = [d.get("mode_share_auto", 0.5) for d in trajectory]
        transit_shares = [d.get("mode_share_transit", 0.4) for d in trajectory]

        fig.add_trace(go.Scatter(
            x=days, y=auto_shares,
            mode="lines", name="Auto",
            line=dict(color="indianred"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=days, y=transit_shares,
            mode="lines", name="Transit",
            line=dict(color="seagreen"),
        ), row=1, col=1)

        # Travel times
        travel_times = [d.get("mean_travel_time", 25) for d in trajectory]
        fig.add_trace(go.Scatter(
            x=days, y=travel_times,
            mode="lines", name="Mean TT",
            line=dict(color="steelblue"),
        ), row=1, col=2)

        # Agent states (final day)
        final_day = trajectory[-1]
        agent_modes = final_day.get("agent_modes", {"auto": 500, "transit": 400, "other": 100})
        fig.add_trace(go.Pie(
            labels=list(agent_modes.keys()),
            values=list(agent_modes.values()),
            name="Final Day",
        ), row=2, col=1)

        # Learning curves (satisfaction over time)
        satisfaction = [d.get("mean_satisfaction", 0.5) for d in trajectory]
        fig.add_trace(go.Scatter(
            x=days, y=satisfaction,
            mode="lines", name="Satisfaction",
            line=dict(color="purple"),
        ), row=2, col=2)

    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_yaxes(title_text="Share", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_yaxes(title_text="Minutes", row=1, col=2)
    fig.update_xaxes(title_text="Day", row=2, col=2)

    fig.update_layout(
        title=f"Agent-Based Model Dynamics: {results.scenario_name}",
        height=600,
    )

    return fig


def plot_stochastic_distribution(
    results: ModelResult,
) -> "go.Figure":
    """Plot stochastic model uncertainty distributions.

    Args:
        results: Results from stochastic model

    Returns:
        Plotly figure with uncertainty visualization
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Travel Time Distribution",
            "Revenue Distribution",
            "Reliability Metrics",
            "Percentile Comparison",
        ],
    )

    # Get replication data
    replications = results.parameters.get("replication_results", [])

    if replications:
        travel_times = [r.get("mean_travel_time", 25) for r in replications]
        revenues = [r.get("toll_revenue", 100000) for r in replications]

        # Travel time histogram
        fig.add_trace(go.Histogram(
            x=travel_times,
            nbinsx=20,
            name="Travel Time",
            marker_color="steelblue",
        ), row=1, col=1)

        # Add percentile lines
        p50 = np.percentile(travel_times, 50)
        p95 = np.percentile(travel_times, 95)
        fig.add_vline(x=p50, line_dash="solid", line_color="green",
                      annotation_text="P50", row=1, col=1)
        fig.add_vline(x=p95, line_dash="dash", line_color="red",
                      annotation_text="P95", row=1, col=1)

        # Revenue histogram
        fig.add_trace(go.Histogram(
            x=revenues,
            nbinsx=20,
            name="Revenue",
            marker_color="seagreen",
        ), row=1, col=2)

        # Reliability box
        reliability_metrics = {
            "Buffer Index": np.std(travel_times) / np.mean(travel_times),
            "Planning Time Index": np.percentile(travel_times, 95) / np.percentile(travel_times, 50),
            "CV": np.std(travel_times) / np.mean(travel_times),
        }

        fig.add_trace(go.Bar(
            x=list(reliability_metrics.keys()),
            y=list(reliability_metrics.values()),
            marker_color="orange",
        ), row=2, col=1)

        # Percentile comparison
        percentiles = [10, 25, 50, 75, 90, 95]
        tt_percentiles = [np.percentile(travel_times, p) for p in percentiles]

        fig.add_trace(go.Scatter(
            x=percentiles,
            y=tt_percentiles,
            mode="lines+markers",
            name="Travel Time Percentiles",
            line=dict(color="steelblue"),
        ), row=2, col=2)

    fig.update_xaxes(title_text="Travel Time (min)", row=1, col=1)
    fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Percentile", row=2, col=2)
    fig.update_yaxes(title_text="Travel Time (min)", row=2, col=2)

    fig.update_layout(
        title=f"Stochastic Model Uncertainty: {results.scenario_name}",
        height=600,
        showlegend=False,
    )

    return fig


def plot_dynamic_evolution(
    results: ModelResult,
) -> "go.Figure":
    """Plot dynamic/Markov model evolution over time.

    Args:
        results: Results from dynamic model

    Returns:
        Plotly figure with dynamic evolution
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "State Evolution",
            "Transition Probabilities",
            "Steady State Convergence",
            "Phase Portrait",
        ],
    )

    trajectory = results.trajectory or []

    if trajectory:
        days = list(range(1, len(trajectory) + 1))

        # State evolution
        states = [d.get("state", {}) for d in trajectory]
        if states and isinstance(states[0], dict):
            for state_name in states[0].keys():
                values = [s.get(state_name, 0) for s in states]
                fig.add_trace(go.Scatter(
                    x=days, y=values,
                    mode="lines", name=state_name,
                ), row=1, col=1)
        else:
            # Scalar state
            fig.add_trace(go.Scatter(
                x=days, y=states,
                mode="lines", name="State",
            ), row=1, col=1)

        # Mode share evolution as proxy
        auto = [d.get("mode_share_auto", 0.5) for d in trajectory]
        transit = [d.get("mode_share_transit", 0.4) for d in trajectory]

        # Transition matrix heatmap (if available)
        trans_matrix = results.parameters.get("transition_matrix")
        if trans_matrix is not None:
            fig.add_trace(go.Heatmap(
                z=trans_matrix,
                colorscale="Blues",
            ), row=1, col=2)

        # Convergence to steady state
        if len(auto) > 1:
            diffs = [abs(auto[i] - auto[i-1]) for i in range(1, len(auto))]
            fig.add_trace(go.Scatter(
                x=days[1:], y=diffs,
                mode="lines", name="Change",
                line=dict(color="orange"),
            ), row=2, col=1)

        # Phase portrait (auto vs transit)
        fig.add_trace(go.Scatter(
            x=auto, y=transit,
            mode="lines+markers",
            name="Trajectory",
            marker=dict(
                size=5,
                color=days,
                colorscale="Viridis",
                showscale=True,
            ),
        ), row=2, col=2)

    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_xaxes(title_text="From State", row=1, col=2)
    fig.update_yaxes(title_text="To State", row=1, col=2)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="|Δ Mode Share|", row=2, col=1)
    fig.update_xaxes(title_text="Auto Share", row=2, col=2)
    fig.update_yaxes(title_text="Transit Share", row=2, col=2)

    fig.update_layout(
        title=f"Dynamic Model Evolution: {results.scenario_name}",
        height=600,
    )

    return fig


def plot_optimization(
    results: ModelResult,
) -> "go.Figure":
    """Plot optimization model results.

    Args:
        results: Results from optimization model

    Returns:
        Plotly figure with optimization results
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Objective Function Evolution",
            "Optimal Toll Schedule",
            "Pareto Frontier",
            "Constraint Satisfaction",
        ],
    )

    # Objective evolution
    fitness_history = results.parameters.get("fitness_history", [])
    if fitness_history:
        gens = list(range(1, len(fitness_history) + 1))
        fig.add_trace(go.Scatter(
            x=gens, y=fitness_history,
            mode="lines", name="Best Fitness",
            line=dict(color="steelblue"),
        ), row=1, col=1)

    # Optimal toll schedule
    optimal_toll = results.parameters.get("optimal_toll", 15)
    toll_by_period = results.parameters.get("optimal_tolls_by_period", {
        "AM Peak": optimal_toll * 1.2,
        "Midday": optimal_toll * 0.7,
        "PM Peak": optimal_toll,
        "Evening": optimal_toll * 0.5,
    })

    fig.add_trace(go.Bar(
        x=list(toll_by_period.keys()),
        y=list(toll_by_period.values()),
        marker_color="seagreen",
    ), row=1, col=2)

    # Pareto frontier (revenue vs travel time)
    pareto_points = results.parameters.get("pareto_frontier", [])
    if pareto_points:
        revenues = [p.get("revenue", 0) for p in pareto_points]
        travel_times = [p.get("travel_time", 25) for p in pareto_points]

        fig.add_trace(go.Scatter(
            x=revenues, y=travel_times,
            mode="markers+lines",
            name="Pareto Frontier",
            marker=dict(size=8, color="orange"),
        ), row=2, col=1)

    # Constraint satisfaction
    constraints = results.parameters.get("constraint_values", {
        "Max Toll": 0.8,
        "Equity": 0.9,
        "Revenue Min": 1.0,
        "Capacity": 0.95,
    })

    fig.add_trace(go.Bar(
        x=list(constraints.keys()),
        y=list(constraints.values()),
        marker_color=["green" if v <= 1 else "red" for v in constraints.values()],
    ), row=2, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Fitness", row=1, col=1)
    fig.update_xaxes(title_text="Period", row=1, col=2)
    fig.update_yaxes(title_text="Toll ($)", row=1, col=2)
    fig.update_xaxes(title_text="Revenue ($)", row=2, col=1)
    fig.update_yaxes(title_text="Travel Time (min)", row=2, col=1)

    fig.update_layout(
        title=f"Optimization Model: {results.scenario_name}",
        height=600,
        showlegend=False,
    )

    return fig


def plot_game_theory(
    results: ModelResult,
) -> "go.Figure":
    """Plot game theory model results.

    Args:
        results: Results from game theory model

    Returns:
        Plotly figure with game theory visualizations
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Player Utilities",
            "Strategy Distribution",
            "Price of Anarchy",
            "Nash Equilibrium",
        ],
    )

    # Player utilities
    utilities = results.parameters.get("player_utilities", {
        "Commuters": -25,
        "Businesses": -30,
        "Authority": 100000,
    })

    fig.add_trace(go.Bar(
        x=list(utilities.keys()),
        y=list(utilities.values()),
        marker_color=["steelblue", "seagreen", "orange"],
    ), row=1, col=1)

    # Strategy distribution
    strategies = results.parameters.get("strategy_distribution", {
        "Drive Alone": 0.4,
        "Carpool": 0.15,
        "Transit": 0.35,
        "WFH": 0.1,
    })

    fig.add_trace(go.Pie(
        labels=list(strategies.keys()),
        values=list(strategies.values()),
    ), row=1, col=2)

    # Price of Anarchy comparison
    poa = results.parameters.get("price_of_anarchy", 1.25)
    fig.add_trace(go.Bar(
        x=["Social Optimum", "Nash Equilibrium"],
        y=[1.0, poa],
        marker_color=["seagreen", "indianred"],
        text=[f"1.00", f"{poa:.2f}"],
        textposition="outside",
    ), row=2, col=1)

    # Nash equilibrium convergence
    nash_history = results.parameters.get("nash_convergence", [])
    if nash_history:
        iters = list(range(1, len(nash_history) + 1))
        fig.add_trace(go.Scatter(
            x=iters, y=nash_history,
            mode="lines+markers",
            name="Gap to Nash",
        ), row=2, col=2)

    fig.update_yaxes(title_text="Utility", row=1, col=1)
    fig.update_yaxes(title_text="Cost Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Equilibrium Gap", row=2, col=2)

    fig.update_layout(
        title=f"Game Theory Model: {results.scenario_name}",
        height=600,
    )

    return fig


def plot_ml(
    results: ModelResult,
) -> "go.Figure":
    """Plot machine learning model results.

    Args:
        results: Results from ML model

    Returns:
        Plotly figure with ML visualizations
    """
    _check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Feature Importance",
            "Prediction vs Actual",
            "Residual Distribution",
            "Treatment Effect (Causal)",
        ],
    )

    model_metrics = results.parameters.get("model_metrics", {})

    # Feature importance
    importance = model_metrics.get("feature_importance", {
        "toll": 0.35,
        "hour": 0.25,
        "day_of_week": 0.15,
        "weather": 0.10,
        "events": 0.08,
        "capacity": 0.07,
    })

    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    fig.add_trace(go.Bar(
        x=[f[1] for f in sorted_features],
        y=[f[0] for f in sorted_features],
        orientation="h",
        marker_color="steelblue",
    ), row=1, col=1)

    # Prediction scatter
    predictions = model_metrics.get("predictions", np.random.normal(25, 3, 100))
    actuals = model_metrics.get("actuals", predictions + np.random.normal(0, 2, 100))

    fig.add_trace(go.Scatter(
        x=actuals, y=predictions,
        mode="markers",
        marker=dict(size=4, opacity=0.5),
    ), row=1, col=2)

    # 45-degree line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(dash="dash", color="red"),
    ), row=1, col=2)

    # Residuals
    residuals = np.array(predictions) - np.array(actuals)
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        marker_color="orange",
    ), row=2, col=1)

    # Treatment effect
    treatment_effect = model_metrics.get("treatment_effect", -2.5)
    ci_lower = model_metrics.get("ci_lower", treatment_effect - 1)
    ci_upper = model_metrics.get("ci_upper", treatment_effect + 1)

    fig.add_trace(go.Scatter(
        x=["Treatment Effect"],
        y=[treatment_effect],
        mode="markers",
        marker=dict(size=12, color="seagreen"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[ci_upper - treatment_effect],
            arrayminus=[treatment_effect - ci_lower],
        ),
    ), row=2, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_xaxes(title_text="Importance", row=1, col=1)
    fig.update_xaxes(title_text="Actual", row=1, col=2)
    fig.update_yaxes(title_text="Predicted", row=1, col=2)
    fig.update_xaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Effect (min)", row=2, col=2)

    fig.update_layout(
        title=f"Machine Learning Model: {results.scenario_name}",
        height=600,
        showlegend=False,
    )

    return fig


def plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    parameter_name: str = "toll",
) -> "go.Figure":
    """Plot sensitivity analysis results.

    Args:
        sensitivity_df: DataFrame from sensitivity analysis
        parameter_name: Name of varied parameter

    Returns:
        Plotly figure with sensitivity curves
    """
    _check_plotly()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Outcomes vs Parameter", "Elasticities"],
    )

    # Get parameter values and outcomes
    param_col = "value" if "value" in sensitivity_df.columns else parameter_name
    x_values = sensitivity_df[param_col]

    # Plot each outcome
    outcome_cols = [c for c in sensitivity_df.columns if c not in [param_col, "parameter"]]

    for col in outcome_cols:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=sensitivity_df[col],
            mode="lines+markers",
            name=col,
        ), row=1, col=1)

    # Compute and plot elasticities
    if len(x_values) > 1:
        base_x = x_values.iloc[0]
        elasticities = {}

        for col in outcome_cols:
            base_y = sensitivity_df[col].iloc[0]
            final_y = sensitivity_df[col].iloc[-1]
            final_x = x_values.iloc[-1]

            if base_y != 0 and base_x != 0:
                pct_change_y = (final_y - base_y) / base_y
                pct_change_x = (final_x - base_x) / base_x
                if pct_change_x != 0:
                    elasticities[col] = pct_change_y / pct_change_x

        if elasticities:
            fig.add_trace(go.Bar(
                x=list(elasticities.keys()),
                y=list(elasticities.values()),
                marker_color="steelblue",
            ), row=1, col=2)

    fig.update_xaxes(title_text=parameter_name.replace("_", " ").title(), row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Outcome", row=1, col=2)
    fig.update_yaxes(title_text="Elasticity", row=1, col=2)

    fig.update_layout(
        title=f"Sensitivity Analysis: {parameter_name}",
        height=400,
    )

    return fig


def plot_comparison(
    results: list[ModelResult],
    metrics: list[str] | None = None,
) -> "go.Figure":
    """Plot comparison across models/scenarios.

    Args:
        results: List of model results
        metrics: Metrics to compare

    Returns:
        Plotly figure with comparison
    """
    _check_plotly()

    if metrics is None:
        metrics = ["mean_travel_time", "mode_share_auto", "toll_revenue"]

    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
    )

    names = [f"{r.model_name}" for r in results]
    colors = px.colors.qualitative.Set2[:len(results)]

    for i, metric in enumerate(metrics, 1):
        values = [getattr(r.aggregates, metric, 0) for r in results]

        fig.add_trace(go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            showlegend=(i == 1),
        ), row=1, col=i)

    fig.update_layout(
        title="Model Comparison",
        height=400,
        showlegend=False,
    )

    return fig
