"""Dash web UI for Shadow Network: The Shadow War.

Two-player game interface with network visualization,
action controls, and round results display.
"""

from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dash import Dash, html, dcc, callback_context, no_update
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

import numpy as np

from congestion_pricing.game.engine import GameEngine, RoundResult
from congestion_pricing.game.board import ZONE_COLORS, FACILITY_NODES
from congestion_pricing.game.scoring import score_history
from congestion_pricing.game.narrative import round_briefing, game_over_briefing


def create_network_figure(engine: GameEngine, last_result: RoundResult | None = None) -> go.Figure:
    """Create a Plotly figure of the facility network.

    Args:
        engine: The game engine with board state
        last_result: Optional last round result for coloring edges by risk/flow

    Returns:
        Plotly Figure with network graph
    """
    G = engine.state.board

    edge_x, edge_y = [], []
    edge_colors = []
    edge_labels = []
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]["pos"]
        x1, y1 = G.nodes[v]["pos"]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        if last_result and (u, v) in last_result.edge_flows:
            flow = last_result.edge_flows[(u, v)]
            risk = last_result.edge_risks.get((u, v), data["base_risk"])
            edge_labels.append(f"{data['label']}<br>Flow: {flow:.0f} | Risk: {risk:.1f}")
        else:
            edge_labels.append(f"{data['label']}<br>Base risk: {data['base_risk']} | Capacity: {data['capacity']}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#555"),
        hoverinfo="skip",
        showlegend=False,
    ))

    if last_result:
        for u, v, data in G.edges(data=True):
            x0, y0 = G.nodes[u]["pos"]
            x1, y1 = G.nodes[v]["pos"]
            flow = last_result.edge_flows.get((u, v), 0)
            risk = last_result.edge_risks.get((u, v), data["base_risk"])
            width = max(1, min(10, flow / 30))

            if risk > data["base_risk"] * 2:
                color = "#d94a4a"
            elif risk > data["base_risk"] * 1.3:
                color = "#f5a623"
            else:
                color = "#7bc47f"

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                hovertext=f"{data['label']}<br>Agents: {flow:.0f}<br>Risk: {risk:.1f}",
                hoverinfo="text",
                showlegend=False,
            ))

    node_x = [G.nodes[n]["pos"][0] for n in G.nodes()]
    node_y = [G.nodes[n]["pos"][1] for n in G.nodes()]
    node_colors = [ZONE_COLORS.get(G.nodes[n]["zone"], "#999") for n in G.nodes()]
    node_labels = [G.nodes[n]["label"] for n in G.nodes()]
    node_text = [
        f"<b>{G.nodes[n]['label']}</b><br>Zone: {G.nodes[n]['zone']}"
        for n in G.nodes()
    ]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=20, color=node_colors, line=dict(width=2, color="white")),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title="Facility Map",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
    )

    return fig


def create_score_history_figure(engine: GameEngine) -> go.Figure:
    """Create score history chart."""
    history = score_history(engine.state)

    fig = go.Figure()
    if history:
        rounds = [h["round"] for h in history]
        fig.add_trace(go.Scatter(
            x=rounds, y=[h["spy_score"] for h in history],
            name="Spy Master", mode="lines+markers",
            line=dict(color="#4a90d9", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=rounds, y=[h["chief_score"] for h in history],
            name="Security Chief", mode="lines+markers",
            line=dict(color="#d94a4a", width=2),
        ))

    fig.update_layout(
        title="Score History",
        xaxis_title="Round",
        yaxis_title="Agents",
        plot_bgcolor="#16213e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_poa_figure(engine: GameEngine) -> go.Figure:
    """Create Price of Anarchy gauge."""
    history = score_history(engine.state)
    poa = history[-1]["poa"] if history else 1.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=poa,
        title={"text": "Price of Anarchy"},
        gauge=dict(
            axis=dict(range=[1, 2]),
            bar=dict(color="#f5a623"),
            steps=[
                dict(range=[1, 1.1], color="#2d4a22"),
                dict(range=[1.1, 1.3], color="#4a6a22"),
                dict(range=[1.3, 1.6], color="#7a5a22"),
                dict(range=[1.6, 2.0], color="#7a2222"),
            ],
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    return fig


def serve_game(host: str = "127.0.0.1", port: int = 8050, debug: bool = True) -> None:
    """Launch the Shadow Network game dashboard.

    Args:
        host: Host to bind to
        port: Port number
        debug: Enable Dash debug mode
    """
    if not DASH_AVAILABLE:
        raise ImportError("Dash required: pip install dash dash-bootstrap-components")

    engine = GameEngine()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    route_info = engine.get_route_info()
    edge_list = engine.get_edge_list()

    route_sliders = []
    for ri in route_info:
        short_label = " → ".join(ri["path"][:2]) + f" → ... → {ri['path'][-1]}"
        route_sliders.append(dbc.Row([
            dbc.Col(html.Small(short_label, style={"fontSize": "11px"}), width=5),
            dbc.Col(dcc.Slider(
                id=f"route-slider-{ri['index']}",
                min=0, max=engine.state.total_agents,
                step=10, value=0,
                marks={0: "0", engine.state.total_agents: str(engine.state.total_agents)},
                tooltip={"placement": "bottom"},
            ), width=7),
        ], className="mb-1"))

    patrol_sliders = []
    for edge in edge_list:
        eid = f"{edge['src']}-{edge['dst']}"
        patrol_sliders.append(dbc.Row([
            dbc.Col(html.Small(edge["label"], style={"fontSize": "11px"}), width=5),
            dbc.Col(dcc.Slider(
                id=f"patrol-slider-{eid}",
                min=0, max=engine.state.patrol_budget,
                step=1, value=0,
                marks={0: "0", int(engine.state.patrol_budget): str(int(engine.state.patrol_budget))},
                tooltip={"placement": "bottom"},
            ), width=7),
        ], className="mb-1"))

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("SHADOW NETWORK", style={"color": "#4a90d9", "fontWeight": "bold"}), width=6),
            dbc.Col(html.Div(id="round-display", style={"textAlign": "right", "fontSize": "18px"}), width=6),
        ], className="my-3"),

        dbc.Row([
            dbc.Col(html.Div(id="score-cards"), width=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="network-graph", figure=create_network_figure(engine)),
            ], width=6),

            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Spy Master", children=[
                        html.Div([
                            html.H5("Agent Allocation", className="mt-2"),
                            html.P("Distribute agents across infiltration routes:", style={"fontSize": "12px"}),
                            html.Div(route_sliders, style={"maxHeight": "250px", "overflowY": "auto"}),
                            html.Hr(),
                            dbc.Checkbox(id="broker-toggle", label="Hire Broker (25% commission)", value=False, className="mb-2"),
                            dbc.Row([
                                dbc.Col(html.Small("Broker Surge:"), width=4),
                                dbc.Col(dcc.Slider(id="broker-surge", min=1.0, max=3.0, step=0.1, value=1.0, disabled=True), width=8),
                            ]),
                        ], className="p-2"),
                    ]),

                    dbc.Tab(label="Security Chief", children=[
                        html.Div([
                            html.H5("Patrol Deployment", className="mt-2"),
                            html.P(f"Distribute patrol budget ({engine.state.patrol_budget:.0f} units):", style={"fontSize": "12px"}),
                            html.Div(patrol_sliders, style={"maxHeight": "300px", "overflowY": "auto"}),
                        ], className="p-2"),
                    ]),
                ]),

                dbc.Button("RESOLVE ROUND", id="resolve-btn", color="warning", className="w-100 mt-2", size="lg"),
                html.Div(id="validation-errors", className="text-danger mt-2"),
            ], width=6),
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="score-chart", figure=create_score_history_figure(engine)),
            ], width=4),
            dbc.Col([
                dcc.Graph(id="poa-gauge", figure=create_poa_figure(engine)),
            ], width=3),
            dbc.Col([
                html.Div(id="nash-indicator", className="text-center p-3",
                         style={"backgroundColor": "#16213e", "borderRadius": "8px", "height": "200px"}),
            ], width=2),
            dbc.Col([
                html.Div(id="briefing-panel",
                         style={"backgroundColor": "#16213e", "borderRadius": "8px",
                                "padding": "15px", "fontSize": "12px",
                                "whiteSpace": "pre-wrap", "height": "250px", "overflowY": "auto"}),
            ], width=3),
        ]),

        dcc.Store(id="engine-state", data={"round": 0}),

    ], fluid=True, style={"backgroundColor": "#0f3460", "minHeight": "100vh", "padding": "20px"})

    route_slider_inputs = [Input(f"route-slider-{ri['index']}", "value") for ri in route_info]
    patrol_slider_inputs = [Input(f"patrol-slider-{eid['src']}-{eid['dst']}", "value") for eid in edge_list]

    all_slider_states = (
        [State(f"route-slider-{ri['index']}", "value") for ri in route_info]
        + [State(f"patrol-slider-{e['src']}-{e['dst']}", "value") for e in edge_list]
        + [State("broker-toggle", "value"), State("broker-surge", "value")]
    )

    @app.callback(
        Output("broker-surge", "disabled"),
        Input("broker-toggle", "value"),
    )
    def toggle_broker_surge(broker_on):
        return not broker_on

    @app.callback(
        [
            Output("network-graph", "figure"),
            Output("score-chart", "figure"),
            Output("poa-gauge", "figure"),
            Output("nash-indicator", "children"),
            Output("briefing-panel", "children"),
            Output("round-display", "children"),
            Output("score-cards", "children"),
            Output("validation-errors", "children"),
            Output("engine-state", "data"),
        ],
        Input("resolve-btn", "n_clicks"),
        all_slider_states,
        prevent_initial_call=True,
    )
    def resolve_round(n_clicks, *slider_values):
        n_route = len(route_info)
        n_edge = len(edge_list)

        route_vals = list(slider_values[:n_route])
        patrol_vals = list(slider_values[n_route:n_route + n_edge])
        broker_on = slider_values[n_route + n_edge]
        broker_surge_val = slider_values[n_route + n_edge + 1]

        errors = engine.submit_spy_master_plan(
            route_allocations=route_vals,
            use_broker=broker_on,
            broker_surge=broker_surge_val or 1.0,
        )
        if errors:
            return [no_update] * 8 + [no_update]

        edge_patrols = {}
        for i, e in enumerate(edge_list):
            edge_patrols[(e["src"], e["dst"])] = patrol_vals[i] or 0
        errors2 = engine.submit_security_chief_plan(edge_patrols)
        if errors2:
            return [no_update] * 8 + [no_update]

        result = engine.resolve_round()
        briefing_text = round_briefing(result, engine.state)
        result.briefing = briefing_text

        net_fig = create_network_figure(engine, result)
        score_fig = create_score_history_figure(engine)
        poa_fig = create_poa_figure(engine)

        nash_children = [
            html.H4("NASH" if result.is_nash else "NOT NASH",
                     style={"color": "#7bc47f" if result.is_nash else "#f5a623"}),
            html.P(f"Distance: {result.nash_distance:.3f}", style={"fontSize": "12px"}),
            html.P("Equilibrium reached" if result.is_nash else "Players can improve",
                    style={"fontSize": "11px"}),
        ]

        round_text = f"Round {engine.state.current_round}/{engine.state.total_rounds}"
        if engine.state.game_over:
            briefing_text = game_over_briefing(engine.state)
            round_text = f"GAME OVER — {engine.state.winner} wins!"

        score_cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(f"{engine.state.spy_master_cumulative:.0f}", className="text-primary"),
                html.Small("Spy Master (infiltrated)"),
            ]), color="dark"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(f"{engine.state.security_chief_cumulative:.0f}", className="text-danger"),
                html.Small("Security Chief (detected)"),
            ]), color="dark"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(f"{result.price_of_anarchy:.2f}", style={"color": "#f5a623"}),
                html.Small("Price of Anarchy"),
            ]), color="dark"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(f"{result.total_detection_risk:,.0f}", style={"color": "#9b59b6"}),
                html.Small("Total Detection Risk"),
            ]), color="dark"), width=3),
        ])

        return [
            net_fig, score_fig, poa_fig,
            nash_children, briefing_text, round_text, score_cards, "",
            {"round": engine.state.current_round},
        ]

    app.run(host=host, port=port, debug=debug)
