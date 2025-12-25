"""Unified dashboard for congestion pricing model visualization.

This module provides an interactive dashboard that can display
results from any model family with appropriate visualizations.
"""

from pathlib import Path
from typing import Any, Callable

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dash import Dash, html, dcc, callback, Output, Input, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from congestion_pricing.policy.results import ModelResult
from congestion_pricing.visualization.plots import (
    plot_mode_shares,
    plot_travel_time_distribution,
    plot_toll_revenue,
    plot_convergence,
    create_summary_card,
    plot_equity_metrics,
)
from congestion_pricing.visualization.model_plots import (
    plot_demand_response,
    plot_equilibrium,
    plot_abm_trajectory,
    plot_stochastic_distribution,
    plot_dynamic_evolution,
    plot_optimization,
    plot_game_theory,
    plot_ml,
    plot_comparison,
)
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


# Model-specific plot registry
MODEL_PLOTS: dict[str, Callable[[ModelResult], "go.Figure"]] = {
    "demand_response": plot_demand_response,
    "equilibrium": plot_equilibrium,
    "abm": plot_abm_trajectory,
    "stochastic": plot_stochastic_distribution,
    "dynamic": plot_dynamic_evolution,
    "optimization": plot_optimization,
    "game_theory": plot_game_theory,
    "ml": plot_ml,
}


class Dashboard:
    """Unified dashboard for model visualization.

    Provides both static HTML export and interactive Dash server.
    """

    def __init__(
        self,
        results: ModelResult | list[ModelResult],
        title: str = "Congestion Pricing Dashboard",
    ):
        """Initialize dashboard.

        Args:
            results: Single result or list of results
            title: Dashboard title
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required: pip install plotly")

        self.results = results if isinstance(results, list) else [results]
        self.title = title
        self._figures: dict[str, go.Figure] = {}

    def generate_figures(self) -> dict[str, go.Figure]:
        """Generate all figures for the dashboard.

        Returns:
            Dictionary mapping section names to figures
        """
        figures = {}

        # Summary card for first/primary result
        figures["summary"] = create_summary_card(self.results[0])

        # Mode shares comparison
        figures["mode_shares"] = plot_mode_shares(self.results)

        # Travel time for primary result
        figures["travel_time"] = plot_travel_time_distribution(self.results[0])

        # Revenue comparison if multiple results
        if len(self.results) > 1:
            figures["revenue"] = plot_toll_revenue(self.results)

        # Convergence if applicable
        if self.results[0].parameters.get("convergence_history"):
            figures["convergence"] = plot_convergence(self.results[0])

        # Equity metrics
        figures["equity"] = plot_equity_metrics(self.results[0])

        # Model-specific plots
        for result in self.results:
            model_name = result.model_name
            if model_name in MODEL_PLOTS:
                try:
                    fig = MODEL_PLOTS[model_name](result)
                    figures[f"model_{model_name}"] = fig
                except Exception as e:
                    logger.warning(f"Failed to generate {model_name} plot: {e}")

        # Comparison plot if multiple results
        if len(self.results) > 1:
            figures["comparison"] = plot_comparison(self.results)

        self._figures = figures
        return figures

    def create_combined_figure(self) -> go.Figure:
        """Create a single combined figure with all visualizations.

        Returns:
            Combined Plotly figure
        """
        if not self._figures:
            self.generate_figures()

        n_figures = len(self._figures)
        n_cols = 2
        n_rows = (n_figures + 1) // 2

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(self._figures.keys()),
            specs=[[{"type": "xy"}] * n_cols for _ in range(n_rows)],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        for idx, (name, subfig) in enumerate(self._figures.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Add traces from subfigure
            for trace in subfig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            title=self.title,
            height=400 * n_rows,
            showlegend=True,
        )

        return fig

    def to_html(
        self,
        output_path: str | Path,
        include_plotlyjs: bool = True,
    ) -> Path:
        """Export dashboard as static HTML.

        Args:
            output_path: Path for HTML file
            include_plotlyjs: Whether to include Plotly JS

        Returns:
            Path to generated HTML file
        """
        if not self._figures:
            self.generate_figures()

        output_path = Path(output_path)

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }",
            ".dashboard-section { background: #f9f9f9; padding: 15px; border-radius: 8px; }",
            ".full-width { grid-column: span 2; }",
            "h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }",
            "h2 { color: #555; margin-top: 0; }",
            ".metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }",
            ".metric-card { background: white; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }",
            ".metric-label { font-size: 12px; color: #888; }",
            "</style>",
        ]

        if include_plotlyjs:
            html_parts.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')

        html_parts.extend([
            "</head>",
            "<body>",
            f"<h1>{self.title}</h1>",
        ])

        # Summary metrics
        result = self.results[0]
        agg = result.aggregates
        html_parts.extend([
            "<div class='metrics-grid'>",
            f"<div class='metric-card'><div class='metric-value'>{agg.mean_travel_time:.1f}</div><div class='metric-label'>Mean Travel Time (min)</div></div>",
            f"<div class='metric-card'><div class='metric-value'>{agg.mode_share_auto:.1%}</div><div class='metric-label'>Auto Mode Share</div></div>",
            f"<div class='metric-card'><div class='metric-value'>{agg.mode_share_transit:.1%}</div><div class='metric-label'>Transit Mode Share</div></div>",
            f"<div class='metric-card'><div class='metric-value'>${agg.toll_revenue:,.0f}</div><div class='metric-label'>Daily Revenue</div></div>",
            f"<div class='metric-card'><div class='metric-value'>{'Yes' if agg.converged else 'No'}</div><div class='metric-label'>Converged</div></div>",
            f"<div class='metric-card'><div class='metric-value'>{agg.runtime_seconds:.1f}s</div><div class='metric-label'>Runtime</div></div>",
            "</div>",
        ])

        # Plot grid
        html_parts.append("<div class='dashboard-grid'>")

        for name, fig in self._figures.items():
            # Determine if full width
            is_full = name in ["summary", "comparison", "model_equilibrium", "model_abm"]
            width_class = "full-width" if is_full else ""

            plot_html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
            )

            html_parts.extend([
                f"<div class='dashboard-section {width_class}'>",
                f"<h2>{name.replace('_', ' ').title()}</h2>",
                plot_html,
                "</div>",
            ])

        html_parts.extend([
            "</div>",
            "</body>",
            "</html>",
        ])

        output_path.write_text("\n".join(html_parts))
        logger.info(f"Dashboard exported to {output_path}")

        return output_path

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
    ) -> None:
        """Serve interactive dashboard using Dash.

        Args:
            host: Host to bind to
            port: Port number
            debug: Enable debug mode
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash required for interactive dashboard. "
                "Install with: pip install dash dash-bootstrap-components"
            )

        if not self._figures:
            self.generate_figures()

        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )

        # Build layout
        result = self.results[0]
        agg = result.aggregates

        # Metric cards
        metric_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{agg.mean_travel_time:.1f} min", className="card-title"),
                    html.P("Mean Travel Time", className="card-text text-muted"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{agg.mode_share_auto:.1%}", className="card-title"),
                    html.P("Auto Share", className="card-text text-muted"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{agg.mode_share_transit:.1%}", className="card-title"),
                    html.P("Transit Share", className="card-text text-muted"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"${agg.toll_revenue:,.0f}", className="card-title"),
                    html.P("Daily Revenue", className="card-text text-muted"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("✓" if agg.converged else "✗", className="card-title"),
                    html.P("Converged", className="card-text text-muted"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{agg.runtime_seconds:.1f}s", className="card-title"),
                    html.P("Runtime", className="card-text text-muted"),
                ])
            ]), width=2),
        ], className="mb-4")

        # Tabs for different views
        tabs = dbc.Tabs([
            dbc.Tab(label="Overview", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=self._figures.get("mode_shares", go.Figure())),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=self._figures.get("travel_time", go.Figure())),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=self._figures.get("equity", go.Figure())),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=self._figures.get("convergence", go.Figure())),
                    ], width=6) if "convergence" in self._figures else dbc.Col(),
                ]),
            ]),
        ])

        # Add model-specific tabs
        for name, fig in self._figures.items():
            if name.startswith("model_"):
                model_name = name.replace("model_", "").replace("_", " ").title()
                tabs.children.append(
                    dbc.Tab(
                        label=model_name,
                        children=[dcc.Graph(figure=fig)],
                    )
                )

        # Add comparison tab if multiple results
        if len(self.results) > 1 and "comparison" in self._figures:
            tabs.children.append(
                dbc.Tab(
                    label="Comparison",
                    children=[
                        dcc.Graph(figure=self._figures["comparison"]),
                        dcc.Graph(figure=self._figures.get("revenue", go.Figure())),
                    ],
                )
            )

        app.layout = dbc.Container([
            html.H1(self.title, className="my-4"),
            html.Hr(),
            metric_cards,
            tabs,
        ], fluid=True)

        logger.info(f"Starting dashboard at http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)

    def show(self) -> None:
        """Display dashboard in Jupyter notebook."""
        if not self._figures:
            self.generate_figures()

        try:
            from IPython.display import display, HTML

            for name, fig in self._figures.items():
                display(HTML(f"<h3>{name.replace('_', ' ').title()}</h3>"))
                fig.show()

        except ImportError:
            logger.warning("IPython not available. Use to_html() or serve() instead.")


class DashboardBuilder:
    """Builder for constructing custom dashboards."""

    def __init__(self, title: str = "Custom Dashboard"):
        """Initialize builder.

        Args:
            title: Dashboard title
        """
        self.title = title
        self._sections: list[dict[str, Any]] = []
        self._results: list[ModelResult] = []

    def add_result(self, result: ModelResult) -> "DashboardBuilder":
        """Add a result to the dashboard.

        Args:
            result: Model result to add

        Returns:
            Self for chaining
        """
        self._results.append(result)
        return self

    def add_section(
        self,
        name: str,
        figure: go.Figure,
        full_width: bool = False,
    ) -> "DashboardBuilder":
        """Add a custom section.

        Args:
            name: Section name
            figure: Plotly figure
            full_width: Whether to span full width

        Returns:
            Self for chaining
        """
        self._sections.append({
            "name": name,
            "figure": figure,
            "full_width": full_width,
        })
        return self

    def add_mode_shares(self) -> "DashboardBuilder":
        """Add mode shares section."""
        if self._results:
            fig = plot_mode_shares(self._results)
            self.add_section("Mode Shares", fig)
        return self

    def add_travel_times(self) -> "DashboardBuilder":
        """Add travel time section."""
        if self._results:
            fig = plot_travel_time_distribution(self._results[0])
            self.add_section("Travel Time Distribution", fig)
        return self

    def add_model_specific(self) -> "DashboardBuilder":
        """Add model-specific visualizations."""
        for result in self._results:
            if result.model_name in MODEL_PLOTS:
                fig = MODEL_PLOTS[result.model_name](result)
                self.add_section(
                    f"{result.model_name.title()} Analysis",
                    fig,
                    full_width=True,
                )
        return self

    def add_comparison(self) -> "DashboardBuilder":
        """Add comparison section."""
        if len(self._results) > 1:
            fig = plot_comparison(self._results)
            self.add_section("Model Comparison", fig, full_width=True)
        return self

    def build(self) -> Dashboard:
        """Build the dashboard.

        Returns:
            Dashboard instance
        """
        dashboard = Dashboard(self._results, self.title)
        dashboard._figures = {s["name"]: s["figure"] for s in self._sections}
        return dashboard


def create_dashboard(
    results: ModelResult | list[ModelResult],
    output_path: str | Path | None = None,
    serve: bool = False,
    port: int = 8050,
) -> Dashboard | Path:
    """Convenience function to create and display a dashboard.

    Args:
        results: Model results
        output_path: If provided, export to HTML
        serve: If True, start interactive server
        port: Port for interactive server

    Returns:
        Dashboard instance or path to HTML
    """
    dashboard = Dashboard(results)
    dashboard.generate_figures()

    if output_path:
        return dashboard.to_html(output_path)

    if serve:
        dashboard.serve(port=port)
        return dashboard

    return dashboard
