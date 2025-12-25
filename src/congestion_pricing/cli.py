"""Command-line interface for congestion pricing models.

This module provides the Typer CLI application with commands
for running models, managing data, and generating reports.
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="congestion-pricing",
    help="Urban congestion pricing modeling toolkit",
    no_args_is_help=True,
)


@app.command()
def run(
    scenario_file: Path = typer.Argument(
        ...,
        help="Path to scenario YAML file",
        exists=True,
    ),
    model: str = typer.Option(
        "equilibrium",
        "--model", "-m",
        help="Model to run (equilibrium, demand_response, abm, etc.)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for results (defaults to stdout)",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Model configuration YAML file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Run a congestion pricing model on a scenario."""
    import yaml

    from congestion_pricing.models.base import ModelRegistry
    from congestion_pricing.policy.scenario import Scenario
    from congestion_pricing.utils.logging import setup_logging

    setup_logging(verbose=verbose)

    typer.echo(f"Loading scenario from {scenario_file}")

    with open(scenario_file) as f:
        scenario_data = yaml.safe_load(f)

    scenario = Scenario.from_dict(scenario_data)

    # Load model config
    model_config = {}
    if config_file:
        with open(config_file) as f:
            model_config = yaml.safe_load(f)

    # Get model
    try:
        model_class = ModelRegistry.get(model)
    except KeyError:
        typer.echo(f"Unknown model: {model}", err=True)
        typer.echo(f"Available: {list(ModelRegistry.list_models())}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Running {model} model...")

    model_instance = model_class(model_config)
    result = model_instance.run(scenario)

    if output:
        result.save(output)
        typer.echo(f"Results saved to {output}")
    else:
        typer.echo("\n--- Results ---")
        typer.echo(f"Mean travel time: {result.aggregates.mean_travel_time:.2f} min")
        typer.echo(f"Mode share (auto): {result.aggregates.mode_share_auto:.2%}")
        typer.echo(f"Mode share (transit): {result.aggregates.mode_share_transit:.2%}")
        typer.echo(f"Toll revenue: ${result.aggregates.toll_revenue:,.0f}")
        typer.echo(f"Runtime: {result.aggregates.runtime_seconds:.2f}s")


@app.command()
def compare(
    scenario_file: Path = typer.Argument(
        ...,
        help="Path to scenario YAML file",
        exists=True,
    ),
    models: str = typer.Option(
        "equilibrium,demand_response,abm",
        "--models", "-m",
        help="Comma-separated list of models to compare",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for comparison report",
    ),
) -> None:
    """Compare multiple models on a scenario."""
    import yaml

    from congestion_pricing.policy.scenario import Scenario
    from congestion_pricing.evaluation.comparison import (
        compare_models,
        generate_comparison_report,
    )

    typer.echo(f"Loading scenario from {scenario_file}")

    with open(scenario_file) as f:
        scenario_data = yaml.safe_load(f)

    scenario = Scenario.from_dict(scenario_data)

    model_list = [m.strip() for m in models.split(",")]
    typer.echo(f"Comparing models: {model_list}")

    result = compare_models(model_list, scenario)

    report = generate_comparison_report(result, output)

    if not output:
        typer.echo(report)


@app.command()
def sensitivity(
    scenario_file: Path = typer.Argument(
        ...,
        help="Path to scenario YAML file",
        exists=True,
    ),
    model: str = typer.Option(
        "equilibrium",
        "--model", "-m",
        help="Model to run",
    ),
    parameter: str = typer.Option(
        "toll_rate",
        "--parameter", "-p",
        help="Parameter to vary",
    ),
    min_val: float = typer.Option(
        0.0,
        "--min",
        help="Minimum parameter value",
    ),
    max_val: float = typer.Option(
        20.0,
        "--max",
        help="Maximum parameter value",
    ),
    steps: int = typer.Option(
        5,
        "--steps", "-s",
        help="Number of steps",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output CSV path",
    ),
) -> None:
    """Run sensitivity analysis for a parameter."""
    import numpy as np
    import yaml

    from congestion_pricing.models.base import ModelRegistry
    from congestion_pricing.policy.scenario import Scenario
    from congestion_pricing.evaluation.comparison import (
        sensitivity_analysis,
        compute_elasticities,
    )

    typer.echo(f"Loading scenario from {scenario_file}")

    with open(scenario_file) as f:
        scenario_data = yaml.safe_load(f)

    scenario = Scenario.from_dict(scenario_data)

    model_class = ModelRegistry.get(model)
    model_instance = model_class()

    values = np.linspace(min_val, max_val, steps).tolist()
    typer.echo(f"Testing {parameter} from {min_val} to {max_val} ({steps} steps)")

    results = sensitivity_analysis(
        model_instance,
        scenario,
        parameter,
        values,
    )

    if output:
        results.to_csv(output, index=False)
        typer.echo(f"Results saved to {output}")
    else:
        typer.echo("\n--- Sensitivity Results ---")
        typer.echo(results.to_string(index=False))

    # Compute elasticities
    elasticities = compute_elasticities(results)
    if elasticities:
        typer.echo("\n--- Elasticities ---")
        for metric, elast in elasticities.items():
            typer.echo(f"  {metric}: {elast:.3f}")


@app.command()
def download(
    dataset: str = typer.Argument(
        ...,
        help="Dataset to download (london_traffic, nyc_taxi, etc.)",
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output", "-o",
        help="Output directory",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files",
    ),
) -> None:
    """Download datasets for analysis."""
    from congestion_pricing.data.download import DataDownloader

    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = DataDownloader(output_dir)

    typer.echo(f"Downloading {dataset}...")

    try:
        if dataset == "london_traffic":
            path = downloader.download_london_traffic()
        elif dataset == "nyc_taxi":
            path = downloader.download_nyc_taxi()
        elif dataset == "osm":
            city = typer.prompt("Enter city name (e.g., 'Manhattan, New York')")
            path = downloader.download_osm_network(city)
        else:
            typer.echo(f"Unknown dataset: {dataset}", err=True)
            typer.echo("Available: london_traffic, nyc_taxi, osm", err=True)
            raise typer.Exit(1)

        typer.echo(f"Downloaded to {path}")

    except Exception as e:
        typer.echo(f"Download failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    model: str = typer.Argument(
        ...,
        help="Model to validate",
    ),
    data_file: Path = typer.Argument(
        ...,
        help="Path to validation data",
        exists=True,
    ),
    method: str = typer.Option(
        "holdout",
        "--method", "-m",
        help="Validation method (holdout, cross_validate)",
    ),
    folds: int = typer.Option(
        5,
        "--folds", "-k",
        help="Number of folds for cross-validation",
    ),
) -> None:
    """Validate a model against observed data."""
    import yaml

    from congestion_pricing.models.base import ModelRegistry
    from congestion_pricing.policy.scenario import Scenario
    from congestion_pricing.evaluation.validation import (
        holdout_validation,
        cross_validate,
    )

    typer.echo(f"Loading validation data from {data_file}")

    with open(data_file) as f:
        data = yaml.safe_load(f)

    # Expect list of scenarios with observations
    scenarios = [Scenario.from_dict(s) for s in data.get("scenarios", [])]

    if not scenarios:
        typer.echo("No scenarios found in validation data", err=True)
        raise typer.Exit(1)

    model_class = ModelRegistry.get(model)
    model_instance = model_class()

    typer.echo(f"Validating {model} with {method} method")

    if method == "holdout":
        result = holdout_validation(model_instance, scenarios)
    elif method == "cross_validate":
        result = cross_validate(model_class, {}, scenarios, n_folds=folds)
    else:
        typer.echo(f"Unknown method: {method}", err=True)
        raise typer.Exit(1)

    typer.echo(result.summary())


@app.command()
def models():
    """List available models."""
    from congestion_pricing.models.base import ModelRegistry

    # Import all models to register them
    from congestion_pricing import models as _

    typer.echo("Available models:\n")

    for name in sorted(ModelRegistry.list_models()):
        model_class = ModelRegistry.get(name)
        typer.echo(f"  {name}")
        typer.echo(f"    {model_class.description}")
        typer.echo(f"    Version: {model_class.version}")

        features = []
        if model_class.supports_tolls:
            features.append("tolls")
        if model_class.supports_transit:
            features.append("transit")
        if model_class.supports_stochastic:
            features.append("stochastic")
        if model_class.supports_dynamics:
            features.append("dynamics")

        typer.echo(f"    Supports: {', '.join(features)}")
        typer.echo()


@app.command()
def init(
    name: str = typer.Argument(
        "my_scenario",
        help="Name for the new scenario",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output", "-o",
        help="Output directory",
    ),
) -> None:
    """Initialize a new scenario with template files."""
    import yaml

    scenario_dir = output_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Create scenario template
    scenario_template = {
        "name": name,
        "description": "Description of the scenario",
        "toll": {
            "toll_type": "cordon",
            "periods": ["am_peak", "pm_peak", "off_peak"],
            "rates": {
                "am_peak": 15.0,
                "pm_peak": 12.0,
                "off_peak": 5.0,
            },
        },
        "transit": {
            "headway_factor": 1.0,
            "fare_multiplier": 1.0,
        },
        "demand_multiplier": 1.0,
        "data_paths": {
            "network": "data/network.gpkg",
            "zones": "data/zones.geojson",
            "od_matrix": "data/od.csv",
        },
    }

    scenario_path = scenario_dir / "scenario.yaml"
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_template, f, default_flow_style=False)

    # Create model config template
    config_template = {
        "equilibrium": {
            "max_iterations": 100,
            "convergence_threshold": 0.001,
        },
        "abm": {
            "n_agents": 1000,
            "n_days": 100,
        },
        "stochastic": {
            "n_replications": 100,
        },
    }

    config_path = scenario_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_template, f, default_flow_style=False)

    typer.echo(f"Created scenario in {scenario_dir}/")
    typer.echo(f"  - scenario.yaml: Edit scenario parameters")
    typer.echo(f"  - config.yaml: Edit model configurations")


@app.command()
def report(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory with model results",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("report.md"),
        "--output", "-o",
        help="Output report path",
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format (markdown, html, latex)",
    ),
) -> None:
    """Generate a report from model results."""
    import json
    from pathlib import Path

    # Find all result files
    result_files = list(results_dir.glob("*.json"))

    if not result_files:
        typer.echo("No result files found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(result_files)} result files")

    lines = [
        "# Congestion Pricing Model Results",
        "",
        f"## Summary",
        f"Results from {len(result_files)} model runs",
        "",
    ]

    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)

        lines.extend([
            f"### {rf.stem}",
            "",
            f"- Model: {data.get('model_name', 'N/A')}",
            f"- Scenario: {data.get('scenario_name', 'N/A')}",
            "",
        ])

        agg = data.get("aggregates", {})
        if agg:
            lines.extend([
                "**Key Metrics:**",
                "",
                f"- Mean travel time: {agg.get('mean_travel_time', 'N/A')} min",
                f"- Mode share (auto): {agg.get('mode_share_auto', 'N/A')}",
                f"- Toll revenue: {agg.get('toll_revenue', 'N/A')}",
                "",
            ])

    report_content = "\n".join(lines)

    with open(output, "w") as f:
        f.write(report_content)

    typer.echo(f"Report generated: {output}")


@app.command()
def dashboard(
    results_path: Path = typer.Argument(
        ...,
        help="Path to result JSON file or directory with multiple results",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Export dashboard as HTML file",
    ),
    serve: bool = typer.Option(
        False,
        "--serve", "-s",
        help="Start interactive dashboard server",
    ),
    port: int = typer.Option(
        8050,
        "--port", "-p",
        help="Port for dashboard server",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open browser automatically",
    ),
) -> None:
    """Launch visualization dashboard for model results.

    Examples:
        congestion-pricing dashboard results.json
        congestion-pricing dashboard results/ --serve
        congestion-pricing dashboard results.json -o dashboard.html
    """
    import json
    import webbrowser

    from congestion_pricing.policy.results import ModelResult
    from congestion_pricing.visualization.dashboard import Dashboard

    # Load results
    results = []

    if results_path.is_file():
        typer.echo(f"Loading result from {results_path}")
        result = ModelResult.load(results_path)
        results.append(result)
    elif results_path.is_dir():
        result_files = list(results_path.glob("*.json"))
        if not result_files:
            typer.echo("No result files found in directory", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading {len(result_files)} results from {results_path}")
        for rf in result_files:
            try:
                result = ModelResult.load(rf)
                results.append(result)
            except Exception as e:
                typer.echo(f"Warning: Failed to load {rf}: {e}", err=True)
    else:
        typer.echo(f"Path not found: {results_path}", err=True)
        raise typer.Exit(1)

    if not results:
        typer.echo("No valid results loaded", err=True)
        raise typer.Exit(1)

    # Create dashboard
    typer.echo(f"Creating dashboard for {len(results)} result(s)...")
    dash = Dashboard(results)
    dash.generate_figures()

    if output:
        # Export as HTML
        dash.to_html(output)
        typer.echo(f"Dashboard exported to {output}")

        if open_browser:
            webbrowser.open(f"file://{output.absolute()}")

    elif serve:
        # Start interactive server
        typer.echo(f"Starting dashboard server at http://127.0.0.1:{port}")
        typer.echo("Press Ctrl+C to stop")

        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")

        dash.serve(port=port)

    else:
        # Default: export to temp HTML and open
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
        ) as f:
            output_path = Path(f.name)

        dash.to_html(output_path)
        typer.echo(f"Dashboard created at {output_path}")

        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")


@app.command()
def visualize(
    scenario_file: Path = typer.Argument(
        ...,
        help="Path to scenario YAML file",
        exists=True,
    ),
    model: str = typer.Option(
        "equilibrium",
        "--model", "-m",
        help="Model to run and visualize",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Export visualization as HTML",
    ),
    show_all: bool = typer.Option(
        False,
        "--all", "-a",
        help="Run all models and compare",
    ),
) -> None:
    """Run model and visualize results in one step.

    Examples:
        congestion-pricing visualize scenario.yaml --model abm
        congestion-pricing visualize scenario.yaml --all
    """
    import yaml
    import webbrowser
    import tempfile

    from congestion_pricing.models.base import ModelRegistry
    from congestion_pricing.policy.scenario import Scenario
    from congestion_pricing.visualization.dashboard import Dashboard

    typer.echo(f"Loading scenario from {scenario_file}")

    with open(scenario_file) as f:
        scenario_data = yaml.safe_load(f)

    scenario = Scenario.from_dict(scenario_data)

    results = []

    if show_all:
        # Run all available models
        from congestion_pricing import models as _

        model_names = list(ModelRegistry.list_models())
        typer.echo(f"Running {len(model_names)} models...")

        for name in model_names:
            typer.echo(f"  Running {name}...")
            try:
                model_class = ModelRegistry.get(name)
                model_instance = model_class()
                result = model_instance.run(scenario)
                results.append(result)
            except Exception as e:
                typer.echo(f"    Failed: {e}", err=True)
    else:
        # Run single model
        typer.echo(f"Running {model} model...")

        model_class = ModelRegistry.get(model)
        model_instance = model_class()
        result = model_instance.run(scenario)
        results.append(result)

    # Create dashboard
    typer.echo("Generating visualization...")
    dash = Dashboard(results, title=f"Results: {scenario.name}")
    dash.generate_figures()

    # Output
    if output:
        dash.to_html(output)
        typer.echo(f"Visualization saved to {output}")
        webbrowser.open(f"file://{output.absolute()}")
    else:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
        ) as f:
            output_path = Path(f.name)

        dash.to_html(output_path)
        typer.echo(f"Opening visualization...")
        webbrowser.open(f"file://{output_path.absolute()}")


def main():
    """Entry point for the CLI."""
    # Ensure models are registered
    from congestion_pricing import models as _

    app()


if __name__ == "__main__":
    main()
