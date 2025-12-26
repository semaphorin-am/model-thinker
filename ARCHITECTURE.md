# Congestion Pricing Modeling Framework: Architecture Overview

> Reference document for systematic code review

---

## 1. Purpose & Domain

This framework simulates **urban congestion pricing policies**—toll systems that charge vehicles for entering congested areas. It answers policy questions like:

- How will a $15 cordon toll affect mode choice?
- What toll schedule maximizes social welfare?
- Are low-income travelers disproportionately burdened?

**Real-world applications:** London Congestion Charge, NYC CBD Tolling, Singapore ERP

---

## 2. Repository Structure

```
src/congestion_pricing/
├── policy/           # Core domain objects (Scenario, Results)
├── models/           # 9 model implementations
├── network/          # Road graph & routing
├── data/             # Data acquisition & processing
├── evaluation/       # Metrics & validation
├── visualization/    # Plots & dashboard
├── utils/            # Logging, I/O, parallelism
└── cli.py            # Typer CLI application

configs/              # YAML configurations (cities, scenarios)
tests/                # Pytest test suite
```

---

## 3. Core Abstractions

### 3.1 Scenario (`policy/scenario.py`)

The **input specification** for any model run:

```python
@dataclass
class Scenario:
    name: str
    toll: TollSchedule | None      # Pricing policy
    transit: TransitScenario | None # Transit improvements
    demand_multiplier: float        # Scale factor
    data_paths: dict[str, Path]     # Network, OD matrix, etc.
```

**TollSchedule** supports three toll types:
- `cordon`: Flat fee to enter a zone
- `time_varying`: Different rates by time-of-day
- `link_based`: Per-link tolls (e.g., express lanes)

### 3.2 ModelResult (`policy/results.py`)

The **standardized output** from any model:

```python
@dataclass
class ModelResult:
    model_name: str
    scenario_name: str
    aggregates: AggregateResults    # Summary metrics
    link_results: LinkResults       # Per-link flows/times
    od_results: ODResults           # Origin-destination matrices
    trajectory: list[dict]          # Time-series (for dynamic models)
    parameters: dict                # Model-specific outputs
```

**AggregateResults** contains:
- `mean_travel_time`, `mode_share_auto`, `mode_share_transit`
- `toll_revenue`, `emissions_co2_kg`
- `converged`, `runtime_seconds`

### 3.3 BaseModel (`models/base.py`)

Abstract base class enforcing a **common interface**:

```python
class BaseModel(ABC):
    name: str
    version: str
    description: str

    # Feature flags
    supports_tolls: bool
    supports_transit: bool
    supports_stochastic: bool
    supports_dynamics: bool

    @abstractmethod
    def run(self, scenario: Scenario) -> ModelResult:
        """Execute model and return standardized results."""
```

**ModelRegistry** provides dynamic model discovery:
```python
@ModelRegistry.register("equilibrium")
class EquilibriumModel(BaseModel): ...

# Later:
model_class = ModelRegistry.get("equilibrium")
```

---

## 4. The Nine Model Families

| Model | File | Core Algorithm | Key Output |
|-------|------|----------------|------------|
| **demand_response** | `demand_response.py` | Log-linear elasticity | Mode share shifts |
| **equilibrium** | `equilibrium.py` | Frank-Wolfe / BPR | Link flows at UE |
| **optimization** | `optimization.py` | Genetic algorithm | Optimal toll vector |
| **game_theory** | `game_theory.py` | Stackelberg / Nash | Price of Anarchy |
| **abm** | `abm.py` | Agent simulation | Day-by-day trajectory |
| **stochastic** | `stochastic.py` | Monte Carlo | Confidence intervals |
| **dynamic** | `dynamic.py` | Markov chain | Steady-state distribution |
| **ml** | `ml.py` | Gradient boosting / DiD | Predictions, causal effects |
| **systems** | `systems.py` | Multi-model orchestration | Synthesized results |

### Key Algorithms

**Frank-Wolfe (Equilibrium)**
```
1. Initialize flows (all-or-nothing assignment)
2. Repeat:
   a. Compute link travel times via BPR function
   b. Find shortest paths (auxiliary solution)
   c. Move flows toward auxiliary (line search)
   d. Check convergence gap
```

**BPR Travel Time Function**
```
t(v) = t₀ × (1 + α × (v/c)^β)

where:
  t₀ = free-flow time
  v  = flow (vehicles/hour)
  c  = capacity
  α  = 0.15, β = 4.0 (standard parameters)
```

**Agent-Based Model**
```
For each day:
  For each agent:
    1. Observe yesterday's travel time
    2. Update mode utility estimates (learning)
    3. Choose mode (logit with habit persistence)
    4. Experience travel time
  Aggregate and record daily statistics
```

---

## 5. Network Layer (`network/`)

### RoadNetwork (`graph.py`)

Wraps NetworkX DiGraph with transportation-specific methods:

```python
class RoadNetwork:
    def add_edge(self, u, v, length, capacity, free_flow_time): ...
    def get_travel_time(self, u, v, flow): ...  # BPR
    def shortest_path(self, origin, destination): ...
    def all_or_nothing_assignment(self, od_matrix): ...
```

### NetworkBuilder

Factory methods for network construction:
- `from_osm(place_name)`: Download from OpenStreetMap
- `from_geopackage(path)`: Load from GIS file
- `create_grid(rows, cols)`: Synthetic grid
- `create_test_network()`: Braess-style 4-node network

### Zones (`zones.py`)

Traffic Analysis Zones (TAZ) for OD aggregation:
- Centroid connectors to road network
- Zone-to-zone demand matrices
- Geographic operations (contains, intersects)

---

## 6. Data Pipeline (`data/`)

### DataDownloader (`download.py`)

Retrieves datasets with checksums:
- London: TfL traffic counts, TOIDS network
- NYC: TLC taxi trips, LION street network
- OSM: On-demand via osmnx

### DataProcessor (`process.py`)

Transformations:
- OD matrix estimation from trip records
- Network conflation (match GPS to links)
- Temporal aggregation (15-min → hourly)

### Schemas (`schemas.py`)

Pydantic models for data validation:
```python
class LinkRecord(BaseModel):
    link_id: str
    from_node: str
    to_node: str
    length_m: float
    lanes: int
    capacity: int
```

---

## 7. Evaluation (`evaluation/`)

### Metrics (`metrics.py`)

| Metric | Formula | Use Case |
|--------|---------|----------|
| **GEH** | `√(2(M-O)²/(M+O))` | Flow validation (GEH<5 is good) |
| **RMSE** | `√(Σ(pred-actual)²/n)` | General error |
| **MAPE** | `Σ|pred-actual|/actual / n` | Percentage error |
| **R²** | `1 - SS_res/SS_tot` | Explained variance |

### Validation (`validation.py`)

- `holdout_validation()`: Train/test split
- `cross_validate()`: K-fold CV
- `temporal_split()`: Train on past, test on future
- `validate_against_counts()`: Compare to observed data

### Comparison (`comparison.py`)

- `compare_models()`: Run multiple models, tabulate results
- `sensitivity_analysis()`: Vary parameter, observe outputs
- `compute_elasticities()`: Arc elasticity from sensitivity

---

## 8. Visualization (`visualization/`)

### Plot Types (`plots.py`, `model_plots.py`)

**Universal plots:**
- Mode share bars
- Travel time distributions
- Revenue comparisons
- Convergence history
- Equity by income group

**Model-specific:**
- `equilibrium`: V/C histograms, flow scatter
- `abm`: Day-by-day evolution, agent pie charts
- `stochastic`: Uncertainty distributions, reliability
- `optimization`: Pareto frontier, fitness evolution
- `ml`: Feature importance, residual plots

### Dashboard (`dashboard.py`)

```python
dashboard = Dashboard(results)
dashboard.to_html("output.html")   # Static export
dashboard.serve(port=8050)         # Interactive Dash server
```

Layout: Metric cards → Tabbed views by model → Comparison

---

## 9. CLI Interface (`cli.py`)

Built with Typer:

| Command | Purpose |
|---------|---------|
| `run` | Execute single model on scenario |
| `compare` | Run multiple models, generate report |
| `sensitivity` | Parameter sweep analysis |
| `visualize` | Run + dashboard in one step |
| `dashboard` | View saved results |
| `validate` | Cross-validate against data |
| `download` | Fetch datasets |
| `init` | Create scenario template |
| `models` | List available models |

---

## 10. Configuration System

### City Configs (`configs/london.yaml`, `configs/nyc.yaml`)

```yaml
name: london_congestion_charge
bounds: {min_lat: 51.49, max_lat: 51.53, ...}
toll:
  toll_type: cordon
  rates: {weekday_charging: 15.00, ...}
transit:
  modes: [tube, bus, dlr]
value_of_time: {mean: 15.0, std: 7.5}
```

### Model Configs (`configs/models.yaml`)

```yaml
equilibrium:
  max_iterations: 100
  convergence_threshold: 0.001
abm:
  n_agents: 5000
  n_days: 100
  learning_rate: 0.3
```

### Scenario Configs (`configs/scenarios/*.yaml`)

```yaml
name: cordon_15
toll: {toll_type: cordon, rates: {peak: 15.0}}
transit: {headway_factor: 0.9}  # 10% improvement
```

---

## 11. Design Patterns & Principles

| Pattern | Where Used | Benefit |
|---------|------------|---------|
| **Strategy** | Model families | Swap algorithms without changing interface |
| **Registry** | ModelRegistry | Dynamic discovery, extensibility |
| **Builder** | NetworkBuilder, DashboardBuilder | Fluent construction |
| **Dataclass** | Scenario, ModelResult | Immutable value objects |
| **Template Method** | BaseModel.run() | Common flow, customizable steps |

### Key Design Decisions

1. **Scenario-in, Result-out**: All models share I/O contract
2. **Lazy data loading**: Network built on first access
3. **Optional dependencies**: ML/viz extras don't break core
4. **Synthetic fallbacks**: Models work without real data
5. **Parallel-ready**: Stochastic model uses joblib

---

## 12. Testing Strategy

```
tests/
├── conftest.py         # Shared fixtures (sample_scenario, etc.)
├── test_scenario.py    # Serialization, validation
├── test_metrics.py     # GEH, RMSE, MAPE correctness
├── test_models.py      # Each model runs without error
└── test_network.py     # Graph operations, BPR function
```

**Fixtures:**
- `sample_scenario`: Cordon toll with transit
- `no_toll_scenario`: Baseline comparison
- `sample_flows`: Random flow/observed pairs

**Coverage focus:**
- Metric edge cases (zeros, negatives)
- Model convergence flags
- Serialization round-trips

---

## 13. Dependencies

**Core:**
- `numpy`, `pandas`: Numerical computation
- `networkx`: Graph algorithms
- `scipy`: Optimization, statistics
- `geopandas`, `shapely`: Spatial operations
- `typer`, `rich`: CLI interface

**Optional (`[viz]`):**
- `plotly`: Interactive charts
- `dash`: Web dashboard

**Optional (`[ml]`):**
- `scikit-learn`: ML models
- `statsmodels`: Econometrics
- `lightgbm`: Gradient boosting

---

## 14. Review Checklist

When reviewing code, verify:

- [ ] Models return valid `ModelResult` with required fields
- [ ] BPR function handles edge cases (zero capacity, negative flow)
- [ ] Convergence criteria are mathematically sound
- [ ] Serialization preserves all data (round-trip tests)
- [ ] CLI commands have proper error handling
- [ ] Visualization gracefully handles missing data
- [ ] No hardcoded paths or credentials
- [ ] Type hints are accurate and complete
- [ ] Docstrings explain parameters and return values

---

## 15. Extension Points

To add a new model:
1. Create `models/my_model.py`
2. Inherit from `BaseModel`
3. Implement `run(scenario) → ModelResult`
4. Decorate with `@ModelRegistry.register("my_model")`
5. Add visualization in `visualization/model_plots.py`

To add a new city:
1. Create `configs/city.yaml` with bounds, toll structure
2. Add download method to `DataDownloader`
3. Add processing pipeline to `DataProcessor`

---

*Document generated for code review purposes. See `PLAN.md` for original requirements.*
