# Urban Congestion Pricing Modeling Program: Engineering Plan

## Executive Summary

This document specifies an engineering plan for a modular modeling program to analyze urban congestion pricing policies. The program integrates models from multiple methodological families (linear/nonlinear, optimization, game-theoretic, agent-based, network, stochastic, Markov/dynamic, statistical/ML, and complexity/systems) to evaluate policy variants including cordon tolls, time-varying tolls, and link-based tolls. The plan includes data acquisition from publicly available sources, a common interface architecture, and rigorous evaluation protocols.

---

## 1. Problem Specification and Scope

### 1.1 Formal Problem Statement

#### 1.1.1 Decision Variables

The policy authority controls:

- **Toll schedule** `p(t, l)`: monetary charge as a function of time-of-day `t` and optionally link `l`
  - Cordon toll: `p(t)` applied at boundary crossings into a zone
  - Time-varying toll: `p(t)` varying by time period (e.g., peak/off-peak/shoulder)
  - Link-based toll: `p(t, l)` varying by both time and specific road segment

- **Transit investment** `I`: capital and operating expenditure on transit capacity, frequency, or coverage

- **Revenue recycling allocation** `R`: distribution of toll revenues among:
  - Transit subsidy `R_transit`
  - Road maintenance `R_roads`
  - Equity transfers `R_equity` (rebates to low-income travelers)
  - General fund `R_general`

#### 1.1.2 Actors

| Actor | Decision Variables | Objective |
|-------|-------------------|-----------|
| **Travelers** (heterogeneous) | Route `r`, departure time `d`, mode `m`, trip frequency `f` | Minimize generalized cost (time + money + discomfort) |
| **Transit authority** | Service frequency, capacity | Maximize ridership subject to budget |
| **Ride-hail platforms** | Pricing, driver allocation | Maximize profit / market share |
| **Freight operators** | Scheduling, route | Minimize delivery cost |
| **Policy authority** | `p(t,l)`, `I`, `R` | Maximize social welfare subject to constraints |

#### 1.1.3 Constraints

**Hard constraints:**
- Road capacity: flow `x_l <= C_l` (or soft via congestion function)
- Transit capacity: passengers per vehicle-hour bounded
- Budget: `sum(R) <= Revenue(p) + Subsidies`
- Physical network topology

**Policy constraints (scenario-dependent):**
- Revenue target: `Revenue(p) >= R_min`
- Equity bound: welfare loss for bottom income quintile `<= delta`
- Political feasibility: maximum toll level `p_max`

#### 1.1.4 System Dynamics

```
Toll policy p(t,l) --> Generalized cost by mode/route/time
                  --> Traveler choices (route, mode, departure, frequency)
                  --> Link flows x(t,l)
                  --> Congestion / travel times T(x)
                  --> Emissions E(x, speeds)
                  --> Revenue Rev(p, x)
                  --> Feedback to traveler learning (day-to-day)
```

#### 1.1.5 Outcome Variables

- Link flows `x(t, l)`
- Origin-Destination (OD) travel times `T_od(t)`
- Mode shares `s_m` by zone
- Aggregate throughput (person-trips/hour through cordon)
- Emissions proxy `E` (function of vehicle-miles and speed)
- Revenue `Rev`
- Distributional outcomes by income/geography

### 1.2 Baseline Scenario and Policy Variants

#### 1.2.1 Baseline Scenario (No Toll)

- Current road network with observed capacities
- Current transit network from General Transit Feed Specification (GTFS)
- Observed demand patterns (OD matrix by time-of-day)
- No congestion pricing; existing parking costs and fuel taxes only

#### 1.2.2 Policy Variant A: Simple Cordon Toll

- Single toll `p` charged for entering defined central zone
- Fixed rate (not time-varying)
- Revenue allocated to transit operations
- Parameters: toll level `p in {5, 10, 15, 20}` currency units

#### 1.2.3 Policy Variant B: Time-Varying Cordon Toll

- Toll varies by time period:
  - Peak (7-10 AM, 4-7 PM): `p_peak`
  - Shoulder (6-7 AM, 10 AM-4 PM, 7-9 PM): `p_shoulder`
  - Off-peak (other): `p_offpeak` (may be zero)
- Parameters: `(p_peak, p_shoulder, p_offpeak)` triplet

#### 1.2.4 Policy Variant C: Link-Based Toll (Optional Extension)

- Dynamic tolling on specific high-congestion links
- Toll responsive to real-time congestion levels
- Requires more complex modeling and data

### 1.3 Outcome Metrics

#### 1.3.1 Efficiency Metrics

| Metric | Definition | Unit |
|--------|------------|------|
| Mean travel time | `E[T_od]` averaged over OD pairs weighted by demand | minutes |
| Travel time reliability | `std(T_od)` or 90th-percentile travel time | minutes |
| Throughput | Person-trips per hour through cordon | trips/hour |
| Vehicle throughput | Vehicles per hour on key links | vehicles/hour |
| Average speed | Network-wide or cordon-area average | km/h or mph |

#### 1.3.2 Environmental Metrics

| Metric | Definition | Unit |
|--------|------------|------|
| Vehicle-kilometers traveled (VKT) | Total distance driven | km |
| Emissions proxy | `E = sum_l(x_l * d_l * e(v_l))` where `e(v)` is emission rate at speed `v` | kg CO2-eq |
| Mode share (sustainable) | Fraction of trips by transit, walk, bike | proportion |

#### 1.3.3 Financial Metrics

| Metric | Definition | Unit |
|--------|------------|------|
| Gross revenue | `sum_l,t(p(t,l) * x(t,l))` | currency |
| Net revenue | Gross minus collection costs | currency |
| Transit farebox recovery | Fare revenue / operating cost | ratio |

#### 1.3.4 Equity Metrics

| Metric | Definition | Unit |
|--------|------------|------|
| Consumer surplus by income quintile | Change in generalized cost * trips | currency |
| Accessibility change by zone | Jobs reachable within 45 min by income group | jobs |
| Toll burden as % of income | Annual toll paid / median zone income | percentage |
| Geographic equity | Variance in welfare change across zones | currency^2 |
| Schedule flexibility penalty | Extra cost for workers with inflexible schedules | currency |

### 1.4 Decision Horizons

#### 1.4.1 Short Run (Primary Focus)

Travelers adjust:
- **Route choice**: Given OD pair and mode, choose path through network
- **Departure time choice**: Shift within +/- 2 hours of preferred arrival
- **Mode choice**: Car (alone, carpool), transit, ride-hail, bike, walk

Assumptions:
- Residential and work locations fixed
- Vehicle ownership fixed
- Trip frequency fixed at baseline levels

Time scale: Days to weeks for equilibration

#### 1.4.2 Medium Run (Optional Extension)

Travelers additionally adjust:
- **Trip frequency**: Reduce non-essential trips, combine trips
- **Telecommuting**: Substitute remote work for commute trips

Time scale: Months

#### 1.4.3 Long Run (Out of Scope for Core Model)

- Residential/job location changes
- Vehicle ownership changes
- Land use response

These are noted as scenario parameters or sensitivity analyses rather than endogenous model outcomes.

---

## 2. Data Acquisition Plan

### 2.1 Candidate Cities and Datasets

We propose two primary candidate cities with publicly accessible data, plus fallback options.

#### 2.1.1 Primary Candidate A: London, United Kingdom

**Rationale:** London implemented a congestion charge in 2003, providing a natural experiment for causal analysis. Transport for London (TfL) publishes extensive open data.

| Dataset | URL | License | Content |
|---------|-----|---------|---------|
| TfL Unified API | https://api.tfl.gov.uk/ | Open Government Licence v3.0 | Real-time and historical journey times, disruptions |
| TfL Road Network | https://tfl.gov.uk/info-for/open-data-users/our-open-data | OGL v3.0 | Road network, traffic counts, journey time reliability |
| TfL GTFS | https://tfl.gov.uk/info-for/open-data-users/our-open-data#on-this-page-4 | OGL v3.0 | Transit schedules for Tube, buses, rail |
| London Datastore Traffic | https://data.london.gov.uk/dataset/traffic-counts | OGL v3.0 | Annual average daily traffic (AADT) by location |
| UK Census 2021 | https://www.ons.gov.uk/census | Open Government Licence | Population, income proxies, car ownership by LSOA |
| OpenStreetMap UK | https://download.geofabrik.de/europe/great-britain/england/greater-london.html | ODbL | Complete road network geometry |
| Congestion Charge Zone | https://data.london.gov.uk/dataset/congestion-charge-zone | OGL v3.0 | Zone boundary shapefile |

**Access notes:**
- TfL API requires free registration for API key
- Most data downloadable without authentication
- Historical journey time data available via TfL API or Freedom of Information requests

#### 2.1.2 Primary Candidate B: New York City, United States

**Rationale:** NYC is implementing congestion pricing (Central Business District Tolling Program) starting 2024/2025, providing pre/post data opportunity. Extensive open data ecosystem.

| Dataset | URL | License | Content |
|---------|-----|---------|---------|
| NYC Taxi & Limousine Commission (TLC) Trip Data | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page | Public Domain | Pickup/dropoff, times, fares for taxi and for-hire vehicles |
| NYC DOT Traffic Speeds | https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9 | Public Domain | Real-time and historical link speeds |
| NYC DOT Traffic Volume Counts | https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts/btm5-ppia | Public Domain | Intersection and segment counts |
| MTA GTFS | https://new.mta.info/developers | MTA Developer License | Subway, bus, commuter rail schedules |
| NYC LION Street Network | https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt | Public Domain | Official street centerline with attributes |
| American Community Survey | https://data.census.gov/ | Public Domain | Income, commute mode, vehicle ownership by census tract |
| NYC Taxi Zones | https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc | Public Domain | Zone boundaries for aggregation |

**Access notes:**
- All data freely downloadable
- TLC data is large (several GB per year); recommend sampling or focusing on specific months
- DOT traffic speeds updated in near-real-time

#### 2.1.3 Fallback/Supplementary Options

| City | Data Sources | Notes |
|------|--------------|-------|
| Singapore | LTA DataMall (https://datamall.lta.gov.sg/) | Requires registration; has ERP toll data |
| Stockholm | Trafikverket Open Data (https://www.trafikverket.se/tjanster/Oppna_data/) | Congestion tax in place since 2007 |
| San Francisco | DataSF (https://datasf.org/) | Proposed pricing; good transit data |

### 2.2 Minimum Viable Datasets (MVD)

For any city implementation, the following datasets are required:

#### 2.2.1 Road Network Graph

| Attribute | Source | Format | Required Fields |
|-----------|--------|--------|-----------------|
| Topology | OpenStreetMap | .osm.pbf | Nodes, edges, connectivity |
| Geometry | OSM or city data | GeoJSON/Shapefile | Coordinates, length |
| Attributes | OSM + city | Joined | Lanes, speed limit, road class |
| Capacity | Derived | CSV | Estimated from lanes and road class |

**Derivation notes:**
- Capacity estimated as `C = lanes * 1800 veh/hr/lane` for highways, `lanes * 900` for arterials
- Speed limits from OSM `maxspeed` tag or city defaults by road class

#### 2.2.2 Link Speeds or Counts

| Attribute | Priority | Source Options | Temporal Resolution |
|-----------|----------|----------------|---------------------|
| Average speed by link | High | DOT sensors, probe data | 15-min to hourly |
| Traffic counts | High | DOT counts, loop detectors | Hourly or daily |
| Travel time reliability | Medium | TfL/DOT APIs | Derived from speed distributions |

#### 2.2.3 Zone Definitions

| Zone Type | Purpose | Source |
|-----------|---------|--------|
| Traffic Analysis Zones (TAZ) | Demand aggregation | City planning department |
| Taxi zones | Trip aggregation (NYC) | TLC |
| Census tracts/LSOAs | Socioeconomic joins | Census bureau |
| Congestion charge zone | Policy boundary | City transport authority |

#### 2.2.4 Time-of-Day Demand Proxies

| Data Type | Source | Processing |
|-----------|--------|------------|
| Taxi/TNC trip origins | TLC/equivalent | Aggregate by zone and 15-min interval |
| Transit ridership | Turnstile/tap data | Station-level entries by hour |
| Traffic counts | DOT sensors | Directional flow by hour |
| Mobile phone mobility | Academic datasets (e.g., SafeGraph) | Requires DUA; fallback only |

#### 2.2.5 Transit GTFS

| Feed Component | Required | Purpose |
|----------------|----------|---------|
| stops.txt | Yes | Station/stop locations |
| routes.txt | Yes | Service identification |
| trips.txt | Yes | Trip patterns |
| stop_times.txt | Yes | Schedules for travel time computation |
| calendar.txt | Yes | Service patterns |
| frequencies.txt | If available | Headway-based services |
| shapes.txt | Preferred | Route geometry for visualization |

#### 2.2.6 Socioeconomic Geography Proxies

| Variable | Source | Spatial Unit |
|----------|--------|--------------|
| Median household income | Census/ACS | Tract or block group |
| Vehicle ownership | Census/ACS | Tract |
| Commute mode share | Census/ACS | Tract |
| Population density | Census/ACS | Tract |
| Employment density | LEHD/city data | Tract or TAZ |

### 2.3 Ingestion Pipeline Specification

#### 2.3.1 Pipeline Architecture

```
raw/                          # Immutable downloaded files
├── checksums.sha256          # Verification hashes
├── manifest.yaml             # Download dates, URLs, versions
├── osm/
│   └── greater-london-latest.osm.pbf
├── gtfs/
│   └── tfl-gtfs-2024-01.zip
├── traffic/
│   └── dot-speeds-2024.csv
└── census/
    └── acs-2022-5yr.parquet

processed/                    # Derived, reproducible outputs
├── network/
│   ├── nodes.parquet
│   ├── edges.parquet
│   └── graph.gpickle
├── demand/
│   ├── od_matrix_am_peak.parquet
│   └── od_matrix_pm_peak.parquet
├── zones/
│   └── zones.geojson
└── socioeconomic/
    └── tract_attributes.parquet
```

#### 2.3.2 Reproducibility Requirements

| Requirement | Implementation |
|-------------|----------------|
| Checksum verification | SHA-256 hash stored in `checksums.sha256`; verified on load |
| Version pinning | URL with date stamp or version; recorded in `manifest.yaml` |
| Download scripting | `src/data/download.py` with Typer CLI |
| Transformation logging | Each processing step logs inputs, outputs, parameters |
| Environment pinning | `pyproject.toml` with locked dependencies via `uv.lock` |

#### 2.3.3 CLI Commands

```bash
# Download raw data for specified city
python -m congestion_pricing data fetch --city london --datasets osm,gtfs,traffic

# Verify checksums
python -m congestion_pricing data verify --city london

# Build processed datasets
python -m congestion_pricing data process --city london --output processed/

# Generate data quality report
python -m congestion_pricing data report --city london
```

### 2.4 Data Schema and Dictionary

#### 2.4.1 Network Schema

**Table: `edges.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| edge_id | string | Unique edge identifier | "e_12345" |
| from_node | string | Origin node ID | "n_100" |
| to_node | string | Destination node ID | "n_101" |
| length_m | float64 | Edge length in meters | 523.4 |
| lanes | int8 | Number of lanes | 2 |
| speed_limit_kmh | float32 | Posted speed limit | 50.0 |
| road_class | string | OSM highway tag | "primary" |
| capacity_vph | float32 | Estimated capacity (veh/hr) | 1800.0 |
| fft_minutes | float32 | Free-flow travel time | 0.63 |
| geometry | WKB | LineString geometry | ... |
| in_toll_zone | bool | Inside congestion charge area | true |

**Table: `nodes.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| node_id | string | Unique node identifier | "n_100" |
| lat | float64 | Latitude | 51.5074 |
| lon | float64 | Longitude | -0.1278 |
| is_centroid | bool | Zone centroid flag | false |
| zone_id | string | Containing zone (if centroid) | "zone_42" |

#### 2.4.2 Demand Schema

**Table: `od_matrix_{period}.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| origin_zone | string | Origin zone ID | "zone_1" |
| dest_zone | string | Destination zone ID | "zone_42" |
| period | string | Time period identifier | "am_peak" |
| demand_auto | float32 | Auto person-trips | 1234.5 |
| demand_transit | float32 | Transit trips | 2345.6 |
| demand_other | float32 | Walk/bike/other | 456.7 |

#### 2.4.3 Traffic Observations Schema

**Table: `observed_speeds.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| edge_id | string | Edge identifier (joined to network) | "e_12345" |
| timestamp | datetime64 | Observation time | 2024-01-15 08:30:00 |
| speed_kmh | float32 | Observed average speed | 23.4 |
| sample_size | int32 | Number of observations | 45 |

**Table: `observed_counts.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| location_id | string | Count location identifier | "loc_789" |
| edge_id | string | Matched edge (may be null) | "e_12345" |
| timestamp | datetime64 | Observation period start | 2024-01-15 08:00:00 |
| duration_min | int16 | Observation period duration | 60 |
| count_total | int32 | Total vehicle count | 1523 |
| count_heavy | int32 | Heavy vehicle count | 89 |

#### 2.4.4 Socioeconomic Schema

**Table: `zone_attributes.parquet`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| zone_id | string | Zone identifier | "zone_42" |
| population | int32 | Resident population | 5423 |
| median_income | float32 | Median household income | 65000.0 |
| pct_no_vehicle | float32 | % households without car | 0.35 |
| pct_commute_transit | float32 | % commuting by transit | 0.42 |
| employment | int32 | Jobs in zone | 12340 |
| income_quintile | int8 | Income quintile (1-5) | 3 |

### 2.5 Data Limitations and Substitutes

#### 2.5.1 Data We Cannot Obtain Without Cost

| Data Type | Why Unavailable | Proposed Substitute |
|-----------|-----------------|---------------------|
| Individual GPS trajectories | Privacy; commercial | Aggregate taxi/TNC data; synthetic population |
| Real-time toll transaction records | Proprietary | Infer from before/after traffic changes |
| Detailed income by individual | Privacy | Zone-level census proxies |
| Full OD survey | Expensive to conduct | Infer from taxi data + gravity model |
| Stated preference surveys | Requires primary research | Use literature elasticities; sensitivity analysis |

#### 2.5.2 Minimum Viable Path

If full data acquisition is constrained, the following represents the minimum viable dataset:

1. **OpenStreetMap network** (always free)
2. **GTFS from transit agency** (usually free)
3. **Census/ACS socioeconomic data** (free)
4. **Sample traffic counts** (usually published by DOT)
5. **Synthetic OD matrix** (generated via gravity model from census data)

This allows model development and validation against aggregate statistics, with documented uncertainty about demand patterns.

---

## 3. Model Families

Each model family is implemented as a module with a common interface (see Section 4). This section specifies the purpose, inputs/outputs, parameters, computational approach, and validation strategy for each family.

### 3.1 Linear/Nonlinear Demand Response Model

**Module:** `src/models/demand_response.py`

#### 3.1.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Explain / Predict |
| Use case | Back-of-the-envelope sensitivity analysis; quick policy screening |
| Complexity | Low |
| Computation time | Seconds |

#### 3.1.2 Model Specification

The model estimates aggregate demand response to price changes using elasticity-based formulations:

**Linear approximation (local):**
```
Delta_Q = epsilon * Q_0 * (Delta_p / p_0)
```

**Constant elasticity (log-linear):**
```
ln(Q) = alpha + epsilon * ln(p) + beta * X
```

Where:
- `Q` = trip volume (by mode, time period, or zone)
- `p` = generalized cost (time + toll + other monetary costs)
- `epsilon` = price elasticity of demand
- `X` = control variables (income, employment, weather)

**Mode choice (logit approximation):**
```
s_m = exp(V_m) / sum_k(exp(V_k))
V_m = -lambda * (time_m + toll_m / VOT)
```

Where `VOT` = value of time, `lambda` = scale parameter.

#### 3.1.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Baseline demand `Q_0` | Vector by zone/mode/period | OD matrix from data |
| Baseline generalized cost `p_0` | Vector | Network model or data |
| Toll schedule `p(t)` | Vector by period | Policy scenario |
| Elasticity parameters `epsilon` | Scalar or vector | Literature or estimated |

| Output | Type | Description |
|--------|------|-------------|
| Demand change `Delta_Q` | Vector | Change in trips by category |
| Mode share `s_m` | Vector | Proportion by mode |
| Revenue estimate | Scalar | `sum(toll * flow)` |
| Consumer surplus change | Scalar | Approximate welfare impact |

#### 3.1.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Default/Literature Value | Source |
|-----------|---------------------|--------------------------|--------|
| Price elasticity (auto trips) | If time-series available | -0.2 to -0.4 | Goodwin (2004), Litman (2023) |
| Cross-elasticity (auto to transit) | Difficult | 0.05 to 0.15 | Literature |
| Value of time | If SP data available | 50% of wage rate | Standard practice |
| Time elasticity | If data available | -0.5 to -1.0 | Literature |

#### 3.1.5 Computational Approach

1. Load baseline demand and cost vectors
2. Apply elasticity formula for each toll scenario
3. Compute mode shares via logit if mode choice included
4. Aggregate to zone/city-level metrics

**Implementation:** Pure NumPy/Pandas; no solver required.

#### 3.1.6 Validation Strategy

| Validation Type | Method | Acceptance Criterion |
|-----------------|--------|---------------------|
| Historical validation | Compare predicted vs. actual demand changes at London congestion charge introduction (2003) | Within 20% of observed reduction |
| Cross-validation | Fit on pre-period, test on post-period | MAPE < 15% |
| Sensitivity check | Vary elasticity within literature range | Results qualitatively stable |

---

### 3.2 Network Equilibrium Model (Static Assignment)

**Module:** `src/models/equilibrium.py`

#### 3.2.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Predict |
| Use case | Compute route choices and link flows given demand and costs; core traffic assignment |
| Complexity | Medium |
| Computation time | Minutes (depends on network size) |

#### 3.2.2 Model Specification

**Wardrop User Equilibrium (UE):** At equilibrium, all used paths between an OD pair have equal and minimum travel time; no traveler can unilaterally reduce their travel time by switching routes.

**Link performance function (Bureau of Public Roads - BPR):**
```
t_l(x_l) = t_l^0 * [1 + alpha * (x_l / C_l)^beta]
```

Where:
- `t_l^0` = free-flow travel time on link `l`
- `x_l` = flow on link `l`
- `C_l` = capacity of link `l`
- `alpha = 0.15`, `beta = 4` (standard BPR parameters, calibratable)

**Generalized cost with tolls:**
```
c_l(x_l) = t_l(x_l) + toll_l / VOT
```

**Mathematical program (Beckmann formulation):**
```
min sum_l integral_0^{x_l} c_l(w) dw
s.t. sum_{paths p using l} f_p = x_l  for all l
     sum_{paths p for OD} f_p = D_od  for all OD
     f_p >= 0
```

#### 3.2.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Network graph | NetworkX DiGraph | Processed OSM |
| Link attributes | DataFrame | edges.parquet |
| OD demand matrix | DataFrame | od_matrix.parquet |
| Toll schedule | Dict or DataFrame | Policy scenario |
| VOT | Scalar | Assumed or estimated |

| Output | Type | Description |
|--------|------|-------------|
| Link flows `x_l` | Vector | Vehicles per hour by link |
| Link travel times `t_l` | Vector | Minutes by link |
| OD travel times | Matrix | Zone-to-zone times |
| Path flows (optional) | Dict | Flow by path |
| Convergence metrics | Dict | Gap, iterations |

#### 3.2.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Default Value | Calibration Method |
|-----------|---------------------|---------------|-------------------|
| BPR alpha | Yes, if speed-flow data | 0.15 | Regression on observed speed vs. flow |
| BPR beta | Yes, if speed-flow data | 4.0 | Regression |
| Capacity by road class | Partially | HCM defaults | Adjust to match observed flows |
| VOT | Yes, if choice data | $15-25/hr | Mode choice estimation |

#### 3.2.5 Computational Approach

**Algorithm: Frank-Wolfe (convex combinations)**

1. Initialize with all-or-nothing assignment (shortest paths at free-flow)
2. Repeat until convergence:
   a. Compute link costs at current flows
   b. Find shortest paths for all OD pairs (auxiliary solution)
   c. Perform line search to find optimal step size
   d. Update flows as convex combination
   e. Check convergence gap

**Alternative algorithms:**
- Method of Successive Averages (MSA): Simpler, slower convergence
- Origin-Based Algorithm (OBA): Faster for large networks
- Gradient projection: For constrained variants

**Libraries:**
- Custom implementation using NetworkX for graph, NumPy for linear algebra
- Consider `python-igraph` for large networks (faster shortest paths)
- For very large networks: interface to external solver (e.g., SUMO, TransCAD)

#### 3.2.6 Validation Strategy

| Validation Type | Method | Acceptance Criterion |
|-----------------|--------|---------------------|
| Link flow validation | Compare assigned flows to observed counts | GEH < 5 for 85% of links with counts |
| Travel time validation | Compare modeled OD times to observed (GPS/API) | RMSE < 3 min for peak periods |
| Screenline validation | Compare total flow across cordon | Within 10% of observed |
| Convergence check | Relative gap | < 0.01 (1%) |

**GEH statistic:**
```
GEH = sqrt(2 * (M - O)^2 / (M + O))
```
Where M = modeled, O = observed. GEH < 5 is considered acceptable.

---

### 3.3 Optimization Model (Policy Design)

**Module:** `src/models/optimization.py`

#### 3.3.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Prescribe |
| Use case | Find optimal toll schedule, investment allocation, or revenue recycling |
| Complexity | High |
| Computation time | Minutes to hours (depends on formulation) |

#### 3.3.2 Model Specification

**Objective function (social welfare maximization):**
```
max W = Consumer_Surplus + Revenue - External_Costs
      = sum_od [integral D_od^{-1}(q) dq - p_od * q_od]
        + sum_l [toll_l * x_l]
        - sum_l [externality_rate * x_l * d_l]
```

**Simplified objective (minimize total system travel time + externalities):**
```
min Z = sum_l [t_l(x_l) * x_l] + gamma * Emissions(x) - lambda * Equity_Constraint
```

**Decision variables:**
- `p_t` = toll in time period `t` (for time-varying toll)
- `p_l` = toll on link `l` (for link-based toll)
- `I_k` = investment in transit option `k`

**Constraints:**
```
Revenue >= R_min                          (revenue target)
Welfare_loss_Q1 <= delta * Welfare_loss_avg   (equity bound)
p_t <= p_max                              (political feasibility)
sum_k I_k <= Budget                       (budget constraint)
x = UE(p, I)                              (equilibrium constraint)
```

#### 3.3.3 Problem Structure

The problem is a **bilevel optimization** (Mathematical Program with Equilibrium Constraints - MPEC):
- Upper level: Policy authority chooses tolls/investments
- Lower level: Travelers reach equilibrium given policy

**Approaches to solve:**

| Approach | When to Use | Solver |
|----------|-------------|--------|
| Single-level reformulation | Small network, differentiable | IPOPT, KNITRO |
| Iterative sensitivity | Medium network | Custom (gradient-based) |
| Metaheuristic | Large network, non-convex | Genetic algorithm, simulated annealing |
| Simulation-optimization | With ABM | Bayesian optimization |

#### 3.3.4 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Network and demand | As in 3.2 | Data |
| Objective weights | Dict | Policy choice |
| Constraint bounds | Dict | Policy choice |
| Initial toll guess | Vector | Heuristic or zero |

| Output | Type | Description |
|--------|------|-------------|
| Optimal toll schedule | Vector | `p*(t)` or `p*(l,t)` |
| Optimal investment | Vector | `I*` |
| Objective value | Scalar | Welfare measure |
| Binding constraints | List | Which constraints active |
| Sensitivity analysis | Dict | Shadow prices |

#### 3.3.5 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Source |
|-----------|---------------------|--------|
| Externality rate | Literature | $0.05-0.15 per vehicle-km |
| Equity weights | Policy choice | Scenario parameter |
| Value of reliability | If data available | Literature: 1-3x VOT |
| Emission factors | Published data | EPA/DEFRA tables |

#### 3.3.6 Computational Approach

**For convex approximation (fixed demand, no equilibrium):**
```python
import cvxpy as cp

p = cp.Variable(n_periods, nonneg=True)
objective = cp.Minimize(total_cost(p) - revenue(p))
constraints = [revenue(p) >= R_min, p <= p_max]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS)
```

**For bilevel with equilibrium:**
1. Start with initial toll vector `p_0`
2. Solve equilibrium to get flows `x(p)`
3. Compute gradient of objective w.r.t. `p` via sensitivity analysis
4. Update `p` using gradient descent or trust region
5. Repeat until convergence

**Libraries:** cvxpy, scipy.optimize, pyomo (for MILP variants), optuna (for hyperparameter-like search)

#### 3.3.7 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Optimality check | Compare to known solutions on toy networks | Match within tolerance |
| Constraint satisfaction | Verify all constraints met | No violations |
| Sensitivity analysis | Perturb parameters, check result stability | Monotonic responses |
| Reality check | Compare to implemented real-world tolls | Qualitatively similar |

---

### 3.4 Game-Theoretic Layer

**Module:** `src/models/game_theory.py`

#### 3.4.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Explain / Explore |
| Use case | Model strategic interactions; analyze efficiency loss from selfish routing |
| Complexity | Medium-High |
| Computation time | Seconds to minutes |

#### 3.4.2 Model Specification

**Congestion game formulation:**
- Players: `N` travelers (or groups of travelers with same OD)
- Strategies: Routes (paths through network)
- Payoff: Negative of travel time on chosen route

**Nash Equilibrium:** Equivalent to Wardrop UE when players are infinitesimal.

**Price of Anarchy (PoA):**
```
PoA = Total_Cost(Nash) / Total_Cost(Social_Optimum)
```

**Extended actors:**

| Actor | Strategy Space | Objective |
|-------|----------------|-----------|
| Travelers | Route, mode, departure | Minimize personal cost |
| Ride-hail platform | Surge pricing, driver dispatch | Maximize profit |
| Freight operators | Delivery time windows | Minimize cost + penalties |
| Transit authority | Frequency, fares | Maximize ridership or welfare |

**Stackelberg game (leader-follower):**
- Leader: Policy authority sets tolls
- Followers: Travelers respond with equilibrium choices

#### 3.4.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Network, demand | As in 3.2 | Data |
| Player types | List of dicts | Scenario definition |
| Platform parameters | Dict | Assumed or estimated |

| Output | Type | Description |
|--------|------|-------------|
| Equilibrium strategies | Dict by player type | Routes, prices, etc. |
| Price of Anarchy | Scalar | Efficiency loss measure |
| Platform profits | Scalar | If ride-hail modeled |
| Mode shares by player type | Vector | Distributional outcome |

#### 3.4.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Source |
|-----------|---------------------|--------|
| Platform commission rate | Public info | ~25% for Uber/Lyft |
| Driver supply elasticity | Academic estimates | Literature |
| Freight value of time | Industry surveys | 2-3x passenger VOT |
| Schedule delay costs | Academic estimates | Small (1990), literature |

#### 3.4.5 Computational Approach

**For pure congestion game (travelers only):**
- Reduce to Wardrop equilibrium (Section 3.2)

**For multi-actor games:**
1. Fixed-point iteration:
   - Platform sets prices given expected demand
   - Travelers choose mode/route given prices
   - Update demand; repeat until convergence

2. Best-response dynamics:
   - Each actor best-responds to others' current strategies
   - Iterate until no player wants to deviate

**Libraries:** nashpy (for finite games), custom implementation for continuous strategy spaces

#### 3.4.6 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Equilibrium verification | Check no profitable deviations | Tolerance < 0.01 |
| PoA bounds | Compare to theoretical bounds (Roughgarden) | Consistent with theory |
| Behavioral validity | Compare platform mode share to observed | Within 5 pp |

---

### 3.5 Agent-Based / Rule-Based Simulation

**Module:** `src/models/abm.py`

#### 3.5.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Explore |
| Use case | Model heterogeneous behavior, bounded rationality, emergent dynamics |
| Complexity | High |
| Computation time | Minutes to hours |

#### 3.5.2 Model Specification

**Agent representation:**

Each agent `i` has:
- Attributes: home zone, work zone, income, schedule flexibility, VOT
- State: current route, mode, departure time habits
- Behavior rules: route choice, mode choice, learning

**Decision rules:**

| Decision | Rule Type | Description |
|----------|-----------|-------------|
| Route choice | Bounded rational | Choose from top-k routes; probability proportional to exp(-theta * cost) |
| Mode choice | Logit with habit | `P(m) = alpha * P_logit(m) + (1-alpha) * I(m = yesterday's mode)` |
| Departure time | Discrete choice | Choose from slots; trade off schedule delay vs. congestion |
| Learning | Reinforcement | Update expected costs based on experienced outcomes |

**Day-to-day dynamics:**
```
For each day t:
    For each agent i:
        1. Observe current costs (with noise)
        2. Update beliefs about travel times
        3. Make choices based on decision rules
    Aggregate to link flows
    Compute actual travel times from congestion model
    Agents experience outcomes
```

#### 3.5.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Network, base demand | As in 3.2 | Data |
| Agent population | DataFrame | Synthesized from census |
| Behavioral parameters | Dict | Calibrated or literature |
| Number of simulation days | Int | Scenario parameter |

| Output | Type | Description |
|--------|------|-------------|
| Daily link flows | Time series | Flow evolution over days |
| Mode share evolution | Time series | Adaptation trajectory |
| Individual travel logs | DataFrame | Agent-level outcomes |
| Equilibration time | Scalar | Days to reach steady state |
| Distributional outcomes | DataFrame | By income group, zone |

#### 3.5.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Default | Source |
|-----------|---------------------|---------|--------|
| Habit persistence (alpha) | Difficult | 0.7 | Cantillo (2007) |
| Learning rate | Difficult | 0.3 | Assumed |
| Choice set size (k) | Assumed | 3-5 routes | Computational |
| Logit scale (theta) | Yes, from choice data | 0.1 | Estimation |
| VOT distribution | From census income | Lognormal, mean=$20/hr | Census |

#### 3.5.5 Computational Approach

**Population synthesis:**
1. Sample agents from joint distribution of (home zone, work zone, income)
2. Assign attributes based on census marginals
3. Use Iterative Proportional Fitting (IPF) if multiple constraints

**Simulation loop:**
```python
for day in range(n_days):
    # Morning peak
    for agent in agents:
        agent.plan_trip(current_beliefs)

    flows = aggregate_choices(agents)
    travel_times = compute_congestion(flows, network)

    for agent in agents:
        agent.experience_outcome(travel_times)
        agent.update_beliefs()

    log_daily_metrics(day, agents, flows)
```

**Performance optimization:**
- Vectorize agent decisions where possible
- Use sparse matrices for path-link incidence
- Parallelize independent agent computations
- Consider Mesa framework for structure, but custom for performance

**Libraries:** mesa (optional), numpy, numba (for JIT compilation)

#### 3.5.6 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Aggregate flow validation | Compare steady-state flows to observed | GEH < 5 for 80% of links |
| Mode share validation | Compare to observed mode shares | Within 3 pp |
| Dynamics validation | Compare adaptation speed to literature | Order of magnitude match |
| Sensitivity analysis | Vary behavioral params | Results qualitatively stable |

---

### 3.6 Stochastic Extensions

**Module:** `src/models/stochastic.py`

#### 3.6.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Predict (with uncertainty) |
| Use case | Reliability metrics; robustness to shocks; uncertainty quantification |
| Complexity | Medium |
| Computation time | Minutes (Monte Carlo) |

#### 3.6.2 Model Specification

**Sources of uncertainty:**

| Source | Representation | Distribution |
|--------|----------------|--------------|
| Demand variability | Random OD matrix | Poisson or negative binomial by OD |
| Incident/weather shocks | Capacity reduction | Bernoulli occurrence; beta severity |
| Traveler heterogeneity | VOT distribution | Lognormal |
| Measurement error | Observed speeds/counts | Normal with estimated variance |

**Stochastic user equilibrium (SUE):**

Replace deterministic shortest path with random utility:
```
P(route r | OD) = exp(-theta * c_r) / sum_s exp(-theta * c_s)
```

This is a logit-based SUE (also called "probit-based" if using normal errors).

**Monte Carlo framework:**
```
For each replication s in 1..S:
    Draw demand_s from demand distribution
    Draw capacity_s from shock model
    Solve equilibrium(demand_s, capacity_s) -> flows_s, times_s

Compute statistics across replications:
    mean_flow = mean(flows_s)
    std_flow = std(flows_s)
    percentile_time = percentile(times_s, [10, 50, 90])
```

#### 3.6.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Base network, demand | As in 3.2 | Data |
| Demand variance | Scalar or by OD | Estimated from data variability |
| Shock model params | Dict | Incident frequency, severity |
| Number of replications | Int | Typically 100-1000 |

| Output | Type | Description |
|--------|------|-------------|
| Mean and std of flows | Vectors | Uncertainty bands |
| Travel time percentiles | Matrix | 10th, 50th, 90th by OD |
| Reliability metrics | Scalars | Buffer time index, planning time index |
| Value at Risk | Scalar | 95th percentile of cost |
| Robustness measures | Dict | How outcomes change under stress |

#### 3.6.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Source |
|-----------|---------------------|--------|
| Demand CV | Yes, from day-to-day variation | Historical data |
| Incident frequency | Yes, from incident logs | DOT/TfL records |
| Incident severity | Partial | Literature + data |
| Weather effects | Yes, from speed-weather regression | Weather + speed data |

#### 3.6.5 Computational Approach

1. **Demand sampling:**
   ```python
   demand_s = np.random.poisson(base_demand)
   # or for overdispersion:
   demand_s = np.random.negative_binomial(r, p)
   ```

2. **Shock sampling:**
   ```python
   incident_occurs = np.random.binomial(1, p_incident, n_links)
   severity = np.random.beta(a, b, n_links) * incident_occurs
   capacity_s = base_capacity * (1 - severity)
   ```

3. **Parallel Monte Carlo:**
   - Use `joblib` or `multiprocessing` to parallelize replications
   - Each replication is independent

4. **Variance reduction:**
   - Antithetic variates
   - Latin hypercube sampling for input parameters

**Libraries:** numpy, scipy.stats, joblib

#### 3.6.6 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Distribution calibration | Compare simulated travel time distribution to observed | K-S test p > 0.05 |
| Reliability metrics | Compare modeled buffer time to observed | Within 20% |
| Stress test plausibility | Expert review of extreme scenarios | Qualitative reasonableness |

---

### 3.7 Markov / Dynamic Model

**Module:** `src/models/dynamic.py`

#### 3.7.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Predict / Explore |
| Use case | Model day-to-day or week-to-week adaptation as a stochastic process |
| Complexity | Medium-High |
| Computation time | Seconds to minutes |

#### 3.7.2 Model Specification

**State space:**

Aggregate travelers into "choice states" representing current behavior:
- State = (mode, route_cluster, departure_time_bin)
- Example: 10 route clusters x 3 modes x 4 departure bins = 120 states

**Transition dynamics:**

Let `pi_t` = distribution over states at day `t`

```
pi_{t+1} = pi_t * P(costs_t)
```

Where `P(costs_t)` is a transition matrix depending on experienced costs.

**Transition probabilities:**

```
P(state j | state i, costs) =
    (1 - lambda) * I(i = j)  +        # Inertia
    lambda * f(cost_j, costs)          # Switching probability
```

Where `f` is a logit or similar choice model.

**Equilibrium:**

Find stationary distribution `pi*` such that `pi* = pi* * P(costs(pi*))`.

This is a fixed-point problem coupling transition dynamics with congestion costs.

#### 3.7.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Network, costs | As in 3.2 | Data |
| State definition | Dict | Design choice |
| Transition parameters | Dict | Estimated or assumed |
| Initial distribution | Vector | Observed or uniform |

| Output | Type | Description |
|--------|------|-------------|
| State distribution trajectory | Matrix (time x states) | Evolution over days |
| Stationary distribution | Vector | Long-run equilibrium |
| Convergence time | Scalar | Days to reach steady state |
| Transition probabilities | Matrix | For analysis of dynamics |

#### 3.7.4 Parameters: Estimate vs. Assume

| Parameter | Estimate from Data? | Source |
|-----------|---------------------|--------|
| Inertia (1-lambda) | Yes, from panel data if available | Observed switching rates |
| Logit scale | Yes, from choice data | Estimation |
| State aggregation | Design choice | Balance granularity vs. tractability |

#### 3.7.5 Computational Approach

**Direct simulation:**
```python
for t in range(T):
    costs = compute_costs(state_distribution[t])
    P = compute_transition_matrix(costs, params)
    state_distribution[t+1] = state_distribution[t] @ P
```

**Stationary distribution (fixed point):**
```python
def fixed_point(pi):
    costs = compute_costs(pi)
    P = compute_transition_matrix(costs)
    return pi @ P

from scipy.optimize import fixed_point as solve_fp
pi_star = solve_fp(fixed_point, pi_0)
```

**Eigenvalue analysis:**
```python
# Second eigenvalue magnitude indicates convergence rate
eigenvalues = np.linalg.eigvals(P)
convergence_rate = np.sort(np.abs(eigenvalues))[-2]
```

**Libraries:** numpy, scipy.linalg, scipy.optimize

#### 3.7.6 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Steady-state validation | Compare stationary distribution to observed mode/time shares | Chi-squared test |
| Dynamics validation | If before/after data exists, compare adaptation trajectory | RMSE on daily shares |
| Eigenvalue check | Second eigenvalue consistent with observed switching rates | Order of magnitude |

---

### 3.8 Statistical and Machine Learning Models

**Module:** `src/models/ml.py`

#### 3.8.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Predict |
| Use case | Forecast travel times, congestion patterns; feature importance; causal inference |
| Complexity | Variable |
| Computation time | Seconds to hours (training) |

#### 3.8.2 Model Specification

**Prediction targets:**

| Target | Features | Model Type |
|--------|----------|------------|
| Link travel time | Time of day, day of week, lagged speed, weather, events | Gradient boosting, neural net |
| Zone-level trip generation | Land use, population, employment, time | Regression, random forest |
| Mode share | Demographics, service levels, prices | Logit, random forest |
| Congestion occurrence | Historical patterns, calendar, weather | Classification |

**Predictive models:**

| Model | Use Case | Libraries |
|-------|----------|-----------|
| Linear/Ridge regression | Baseline, interpretable | scikit-learn |
| Random Forest | Nonlinear, feature importance | scikit-learn |
| Gradient Boosting (XGBoost, LightGBM) | Best accuracy for tabular | xgboost, lightgbm |
| LSTM / Transformer | Time series with long memory | pytorch |
| Gaussian Process | Uncertainty quantification | GPyTorch, scikit-learn |

**Causal inference:**

| Method | Requirement | Application |
|--------|-------------|-------------|
| Difference-in-Differences (DiD) | Before/after + control group | London congestion charge effect |
| Synthetic Control | Pre-period for multiple units | Construct synthetic "no-toll" London |
| Regression Discontinuity | Sharp cutoff in treatment | Toll zone boundary effects |
| Instrumental Variables | Valid instrument | Price variation for demand elasticity |

#### 3.8.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Feature matrix X | DataFrame | Processed data |
| Target vector y | Series | Observed outcomes |
| Time indices | DatetimeIndex | For train/test split |
| Causal design | Dict | Treatment timing, control units |

| Output | Type | Description |
|--------|------|-------------|
| Predictions y_hat | Series | Point forecasts |
| Prediction intervals | DataFrame | If uncertainty quantified |
| Feature importance | Series | For interpretability |
| Causal estimates | Dict | Treatment effects with standard errors |
| Model performance | Dict | RMSE, MAE, R2, calibration |

#### 3.8.4 Parameters: Estimate vs. Assume

| Parameter | Estimate Method |
|-----------|-----------------|
| Model hyperparameters | Cross-validation |
| Feature selection | Lasso, recursive feature elimination |
| Causal parameters | DiD regression, synthetic control weights |

#### 3.8.5 Computational Approach

**Training pipeline:**
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', lgb.LGBMRegressor())
])

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipeline.predict(X.iloc[val_idx])
    evaluate(y.iloc[val_idx], y_pred)
```

**Causal inference (DiD example):**
```python
import statsmodels.formula.api as smf

# DiD regression
model = smf.ols('outcome ~ treated * post + C(time) + C(zone)', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['zone']})
treatment_effect = results.params['treated:post']
```

**Libraries:** scikit-learn, statsmodels, xgboost, lightgbm, pytorch (optional), causalml (optional)

#### 3.8.6 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Holdout performance | Test on future time periods | RMSE < baseline |
| Calibration | Reliability diagram for prediction intervals | Coverage close to nominal |
| Feature importance stability | Bootstrap feature importance | Stable ranking |
| Causal validity | Parallel trends test (DiD) | Pre-trends not significant |
| Placebo tests | Run analysis on fake treatment times | No spurious effects |

---

### 3.9 Complexity/Systems Synthesis

**Module:** `src/models/systems.py`

#### 3.9.1 Purpose

| Dimension | Value |
|-----------|-------|
| Primary purpose | Explore / Explain |
| Use case | Integrate feedback loops; scenario analysis; identify leverage points |
| Complexity | Conceptual (simple computation) |
| Computation time | Seconds |

#### 3.9.2 Model Specification

**System dynamics representation:**

Capture feedback loops using stock-and-flow or causal loop diagrams:

**Key feedback loops:**

| Loop | Type | Description |
|------|------|-------------|
| Congestion-route-choice | Balancing | High congestion -> route shifts -> congestion decreases |
| Induced demand | Reinforcing | Lower congestion -> more trips -> congestion returns |
| Transit crowding | Balancing | Mode shift to transit -> crowding -> some shift back |
| Revenue-investment | Reinforcing | Toll revenue -> transit investment -> mode shift -> more through tolled zone |
| Political feedback | Balancing | High tolls -> public opposition -> toll reduction |

**Causal loop diagram (text representation):**
```
Toll Level --(+)--> Generalized Cost
Generalized Cost --(−)--> Auto Trips
Auto Trips --(+)--> Congestion
Congestion --(+)--> Generalized Cost  [BALANCING LOOP: Route Equilibrium]
Auto Trips --(−)--> Transit Trips
Transit Trips --(+)--> Transit Crowding
Transit Crowding --(+)--> Transit Generalized Cost  [BALANCING LOOP: Transit Capacity]
Toll Level --(+)--> Revenue
Revenue --(+)--> Transit Investment
Transit Investment --(−)--> Transit Generalized Cost  [REINFORCING LOOP: Investment Cycle]
```

**Scenario dimensions:**

| Dimension | Low | Medium | High |
|-----------|-----|--------|------|
| Induced demand elasticity | 0 (no new trips) | 0.2 | 0.5 |
| Transit investment fraction | 0.25 | 0.5 | 0.75 |
| Political constraint (max toll) | $5 | $15 | $25 |
| Work-from-home adoption | 10% | 20% | 40% |

#### 3.9.3 Inputs and Outputs

| Input | Type | Source |
|-------|------|--------|
| Model outputs from 3.1-3.8 | Various | Other modules |
| Scenario parameters | Dict | Policy scenarios |
| Feedback loop structure | Diagram | Expert specification |

| Output | Type | Description |
|--------|------|-------------|
| Systems map | Visual | Causal loop diagram |
| Cross-model comparison | DataFrame | Outcome metrics by model |
| Scenario matrix | DataFrame | Outcomes under different assumptions |
| Leverage point analysis | Text | Which parameters matter most |
| Uncertainty characterization | Text | Where models agree/disagree |

#### 3.9.4 Computational Approach

**Model orchestration:**
```python
def run_scenario(scenario_params):
    """Run all model families on same scenario."""
    results = {}

    # Quick screening
    results['demand_response'] = demand_model.run(scenario_params)

    # Equilibrium
    results['equilibrium'] = equilibrium_model.run(scenario_params)

    # Optimization (if prescriptive)
    if scenario_params.get('optimize_toll'):
        results['optimal_toll'] = optimization_model.run(scenario_params)

    # ABM for dynamics
    results['abm'] = abm_model.run(scenario_params)

    # Stochastic for reliability
    results['stochastic'] = stochastic_model.run(scenario_params)

    return aggregate_results(results)
```

**Cross-model validation:**
```python
def compare_models(results):
    """Check consistency across models."""
    metrics = ['mean_travel_time', 'mode_share_transit', 'revenue']

    comparison = pd.DataFrame({
        model_name: {m: res[m] for m in metrics}
        for model_name, res in results.items()
    })

    # Flag large discrepancies
    cv = comparison.std(axis=1) / comparison.mean(axis=1)
    discrepancies = cv[cv > 0.2]

    return comparison, discrepancies
```

**Libraries:** networkx (for graph visualization of systems), matplotlib/plotly (for diagrams)

#### 3.9.5 Validation Strategy

| Validation Type | Method | Criterion |
|-----------------|--------|-----------|
| Internal consistency | Models produce similar aggregate outcomes | CV < 20% on key metrics |
| Face validity | Expert review of systems map | No missing critical loops |
| Scenario plausibility | Extreme scenarios produce extreme outcomes | Monotonic responses |
| Historical scenario | Replicate known policy change (e.g., London 2003) | Direction and magnitude match |

---

## 4. Common Interfaces and Evaluation

This section defines the shared abstractions and evaluation protocols that enable consistent comparison across model families.

### 4.1 Scenario Object

All models consume a common `Scenario` object that encapsulates the inputs for a single model run.

#### 4.1.1 Scenario Schema

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import networkx as nx

@dataclass
class TollSchedule:
    """Toll specification."""
    toll_type: str  # "cordon", "time_varying", "link_based"
    periods: List[str]  # e.g., ["am_peak", "pm_peak", "off_peak"]
    rates: Dict[str, float]  # period -> rate, or (period, link) -> rate
    zone_boundary: Optional[str] = None  # GeoJSON path for cordon

@dataclass
class TransitScenario:
    """Transit service levels."""
    gtfs_path: str
    frequency_multiplier: float = 1.0  # Scale headways
    fare_multiplier: float = 1.0
    capacity_multiplier: float = 1.0

@dataclass
class ShockModel:
    """Stochastic shock specification."""
    incident_probability: float = 0.02  # Per link per day
    severity_alpha: float = 2.0  # Beta distribution param
    severity_beta: float = 5.0
    demand_cv: float = 0.1  # Coefficient of variation

@dataclass
class Scenario:
    """Complete scenario specification for model runs."""
    # Identifiers
    name: str
    description: str

    # Network
    network_path: str  # Path to processed network files
    network: Optional[nx.DiGraph] = None  # Loaded graph

    # Demand
    od_matrix_path: str
    demand_multiplier: float = 1.0

    # Policy
    toll: Optional[TollSchedule] = None
    transit: Optional[TransitScenario] = None
    investment_budget: float = 0.0

    # Behavioral parameters
    vot_mean: float = 20.0  # $/hour
    vot_std: float = 10.0

    # Stochastic
    shocks: Optional[ShockModel] = None
    n_replications: int = 1

    # Simulation parameters
    time_periods: List[str] = field(default_factory=lambda: ["am_peak", "pm_peak"])
    simulation_days: int = 100  # For ABM/dynamic models

    # Metadata
    city: str = "london"
    year: int = 2024

    def load(self):
        """Load data files into memory."""
        # Implementation loads network, OD matrix, etc.
        pass
```

#### 4.1.2 Scenario Factory

```python
class ScenarioFactory:
    """Create standard scenarios for comparison."""

    @staticmethod
    def baseline(city: str) -> Scenario:
        """No-toll baseline."""
        return Scenario(
            name=f"{city}_baseline",
            description="No congestion pricing",
            network_path=f"processed/{city}/network/",
            od_matrix_path=f"processed/{city}/demand/od_matrix.parquet",
            city=city
        )

    @staticmethod
    def simple_cordon(city: str, toll_rate: float) -> Scenario:
        """Fixed cordon toll."""
        base = ScenarioFactory.baseline(city)
        base.name = f"{city}_cordon_{toll_rate}"
        base.toll = TollSchedule(
            toll_type="cordon",
            periods=["all_day"],
            rates={"all_day": toll_rate},
            zone_boundary=f"processed/{city}/zones/cordon.geojson"
        )
        return base

    @staticmethod
    def time_varying(city: str, peak: float, shoulder: float, off: float) -> Scenario:
        """Time-varying cordon toll."""
        base = ScenarioFactory.baseline(city)
        base.name = f"{city}_timevar_{peak}_{shoulder}_{off}"
        base.toll = TollSchedule(
            toll_type="time_varying",
            periods=["am_peak", "shoulder", "pm_peak", "off_peak"],
            rates={
                "am_peak": peak,
                "shoulder": shoulder,
                "pm_peak": peak,
                "off_peak": off
            }
        )
        return base
```

### 4.2 Standard Outputs

All models produce outputs conforming to a common schema, enabling apples-to-apples comparison.

#### 4.2.1 Model Result Schema

```python
@dataclass
class LinkResults:
    """Link-level outputs."""
    edge_id: pd.Series  # str
    period: pd.Series  # str
    flow_vph: pd.Series  # float, vehicles per hour
    travel_time_min: pd.Series  # float
    speed_kmh: pd.Series  # float
    volume_capacity_ratio: pd.Series  # float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'edge_id': self.edge_id,
            'period': self.period,
            'flow_vph': self.flow_vph,
            'travel_time_min': self.travel_time_min,
            'speed_kmh': self.speed_kmh,
            'v_c_ratio': self.volume_capacity_ratio
        })

@dataclass
class ODResults:
    """OD-level outputs."""
    origin: pd.Series
    destination: pd.Series
    period: pd.Series
    travel_time_auto: pd.Series  # minutes
    travel_time_transit: pd.Series
    demand_auto: pd.Series  # trips
    demand_transit: pd.Series
    generalized_cost_auto: pd.Series  # $ equivalent
    generalized_cost_transit: pd.Series

@dataclass
class AggregateResults:
    """City/zone aggregate metrics."""
    # Efficiency
    mean_travel_time: float  # minutes, demand-weighted
    travel_time_p90: float  # 90th percentile
    total_vkt: float  # vehicle-km traveled
    throughput_cordon: float  # trips/hour through charging zone

    # Mode share
    mode_share_auto: float
    mode_share_transit: float
    mode_share_other: float

    # Financial
    toll_revenue: float  # daily $
    transit_fare_revenue: float

    # Environmental
    emissions_co2_kg: float

    # Equity (by income quintile)
    welfare_change_q1: float  # lowest income
    welfare_change_q2: float
    welfare_change_q3: float
    welfare_change_q4: float
    welfare_change_q5: float  # highest income

    # Metadata
    converged: bool
    iterations: int
    runtime_seconds: float

@dataclass
class ModelResult:
    """Complete model output."""
    scenario: Scenario
    model_name: str
    model_version: str
    timestamp: str

    # Detailed results
    link_results: Optional[LinkResults] = None
    od_results: Optional[ODResults] = None

    # Aggregate results
    aggregates: AggregateResults = None

    # For stochastic models
    aggregate_std: Optional[AggregateResults] = None  # Standard deviations
    aggregate_percentiles: Optional[Dict[int, AggregateResults]] = None  # 10, 50, 90

    # For dynamic models
    trajectory: Optional[pd.DataFrame] = None  # Daily aggregates over time

    def to_dict(self) -> dict:
        """Serialize for storage."""
        pass

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelResult':
        """Deserialize from storage."""
        pass
```

#### 4.2.2 Output Storage

```
results/
├── {scenario_name}/
│   ├── {model_name}/
│   │   ├── metadata.json       # Model version, runtime, parameters
│   │   ├── aggregates.json     # AggregateResults
│   │   ├── link_results.parquet
│   │   ├── od_results.parquet
│   │   └── trajectory.parquet  # For dynamic models
│   └── comparison.json         # Cross-model comparison for this scenario
└── summary/
    └── all_scenarios.parquet   # Aggregates across all scenarios and models
```

### 4.3 Evaluation Suite

#### 4.3.1 Holdout Validation Design

**Temporal holdout:**
- Training period: All data except last 3 months
- Validation period: Last 3 months
- Test period: Post-policy period (if available)

**Spatial holdout (for ML models):**
- Training zones: 80% of zones
- Test zones: 20% held out
- Ensure geographic spread in test set

**Cross-validation protocol:**
```python
from sklearn.model_selection import TimeSeriesSplit

def temporal_cv(data, n_splits=5):
    """Time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(data):
        yield data.iloc[train_idx], data.iloc[val_idx]
```

#### 4.3.2 Goodness-of-Fit Metrics

**For flow validation:**

| Metric | Formula | Target |
|--------|---------|--------|
| GEH | `sqrt(2*(M-O)^2/(M+O))` | < 5 for 85% of links |
| RMSE | `sqrt(mean((M-O)^2))` | Minimize |
| %RMSE | `RMSE / mean(O) * 100` | < 25% |
| R-squared | `1 - SS_res/SS_tot` | > 0.8 |

**For travel time validation:**

| Metric | Formula | Target |
|--------|---------|--------|
| MAE | `mean(|M-O|)` | < 3 minutes |
| MAPE | `mean(|M-O|/O) * 100` | < 15% |
| Bias | `mean(M-O)` | Near 0 |

**For mode share validation:**

| Metric | Formula | Target |
|--------|---------|--------|
| Absolute error | `|M-O|` | < 3 percentage points |
| Chi-squared | `sum((M-O)^2/E)` | p > 0.05 |

#### 4.3.3 Policy Counterfactual Validation

When historical policy changes exist (e.g., London 2003), validate:

| Check | Method | Criterion |
|-------|--------|-----------|
| Direction | Did model predict reduction in traffic? | Correct sign |
| Magnitude | Compare % change to observed | Within factor of 2 |
| Timing | For dynamic models, compare adaptation speed | Order of magnitude |

```python
def validate_counterfactual(model_result, observed_change):
    """Compare model prediction to observed policy impact."""
    predicted = model_result.aggregates.throughput_cordon
    baseline = baseline_result.aggregates.throughput_cordon
    predicted_change = (predicted - baseline) / baseline

    actual_change = observed_change  # e.g., -0.18 for 18% reduction

    return {
        'predicted_change': predicted_change,
        'actual_change': actual_change,
        'direction_correct': np.sign(predicted_change) == np.sign(actual_change),
        'magnitude_ratio': predicted_change / actual_change if actual_change != 0 else np.inf
    }
```

#### 4.3.4 Evaluation Pipeline

```python
class EvaluationSuite:
    """Run all validation checks."""

    def __init__(self, observed_data: dict, model_results: List[ModelResult]):
        self.observed = observed_data
        self.results = model_results

    def run_flow_validation(self) -> pd.DataFrame:
        """Compare modeled flows to observed counts."""
        metrics = []
        for result in self.results:
            merged = result.link_results.to_dataframe().merge(
                self.observed['counts'],
                on=['edge_id', 'period']
            )
            geh = compute_geh(merged['flow_vph'], merged['observed_count'])
            metrics.append({
                'model': result.model_name,
                'geh_mean': geh.mean(),
                'geh_pct_under_5': (geh < 5).mean(),
                'rmse': compute_rmse(merged['flow_vph'], merged['observed_count'])
            })
        return pd.DataFrame(metrics)

    def run_travel_time_validation(self) -> pd.DataFrame:
        """Compare modeled travel times to observed."""
        # Similar structure
        pass

    def run_mode_share_validation(self) -> pd.DataFrame:
        """Compare modeled mode shares to survey data."""
        pass

    def run_all(self) -> dict:
        """Run complete evaluation suite."""
        return {
            'flow_validation': self.run_flow_validation(),
            'travel_time_validation': self.run_travel_time_validation(),
            'mode_share_validation': self.run_mode_share_validation()
        }
```

### 4.4 Model Comparison Dashboard

#### 4.4.1 Dashboard Specification

The dashboard provides visual and tabular comparison of model outputs across scenarios.

**Views:**

| View | Content | Visualization |
|------|---------|---------------|
| Scenario Summary | Key metrics for one scenario across all models | Bar chart with error bars |
| Model Comparison | One model across all scenarios | Line chart or heatmap |
| Sensitivity | One metric vs. toll level | Multi-line chart by model |
| Equity | Welfare change by income quintile | Grouped bar chart |
| Convergence | Model agreement/disagreement | Coefficient of variation heatmap |
| Validation | Goodness-of-fit metrics | Table with conditional formatting |

#### 4.4.2 Key Comparison Tables

**Table: Aggregate Outcomes by Model**

| Metric | Demand Response | Equilibrium | ABM | Stochastic | ML |
|--------|-----------------|-------------|-----|------------|-----|
| Mean travel time (min) | 25.3 | 24.1 | 24.5 +/- 0.3 | 24.2 +/- 1.1 | 24.0 |
| Mode share transit (%) | 42.1 | 43.5 | 43.2 | 43.4 | 43.8 |
| Daily revenue ($M) | 1.23 | 1.18 | 1.20 | 1.19 | 1.21 |
| CO2 reduction (%) | -12.3 | -11.8 | -11.5 | -11.7 | -12.0 |
| Welfare Q1 ($) | -45 | -42 | -40 | -41 | -43 |

**Table: Model Validation Summary**

| Model | Flow GEH<5 (%) | TT MAPE (%) | Mode Err (pp) | Counterfactual OK |
|-------|----------------|-------------|---------------|-------------------|
| Equilibrium | 87% | 12% | 2.1 | Yes |
| ABM | 82% | 14% | 1.8 | Yes |
| ML | 91% | 8% | 2.5 | N/A |

#### 4.4.3 Key Plots

**Plot 1: Toll-Response Curve**
- X-axis: Toll level ($)
- Y-axis: Cordon throughput (% of baseline)
- Lines: One per model
- Shows where models agree/diverge

**Plot 2: Equity Impact by Model**
- X-axis: Income quintile (1-5)
- Y-axis: Welfare change ($)
- Grouped bars: One color per model
- Error bars for stochastic/ABM

**Plot 3: Time-of-Day Pattern**
- X-axis: Hour of day
- Y-axis: Cordon flow
- Lines: Baseline vs. policy, observed vs. modeled

**Plot 4: Convergence Diagnostic**
- Heatmap: Rows = metrics, Columns = scenarios
- Color: Coefficient of variation across models
- Highlights where models disagree

#### 4.4.4 Dashboard Implementation

```python
# notebooks/dashboard.py or streamlit app

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def load_results(scenario_name: str) -> List[ModelResult]:
    """Load all model results for a scenario."""
    pass

def plot_toll_response(results_by_toll: Dict[float, List[ModelResult]]):
    """Plot throughput vs toll level by model."""
    fig = go.Figure()
    for model_name in ['equilibrium', 'abm', 'demand_response']:
        tolls = sorted(results_by_toll.keys())
        throughputs = [
            next(r.aggregates.throughput_cordon
                 for r in results_by_toll[t] if r.model_name == model_name)
            for t in tolls
        ]
        fig.add_trace(go.Scatter(x=tolls, y=throughputs, name=model_name))
    fig.update_layout(xaxis_title="Toll ($)", yaxis_title="Throughput (trips/hr)")
    return fig

def create_comparison_table(results: List[ModelResult]) -> pd.DataFrame:
    """Create cross-model comparison table."""
    rows = []
    for r in results:
        rows.append({
            'Model': r.model_name,
            'Mean TT': r.aggregates.mean_travel_time,
            'Mode Share Transit': r.aggregates.mode_share_transit,
            'Revenue': r.aggregates.toll_revenue,
            'Welfare Q1': r.aggregates.welfare_change_q1
        })
    return pd.DataFrame(rows)
```

---

## 5. Repository and Engineering Plan

### 5.1 Repository Layout

```
congestion-pricing/
├── README.md
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies (via uv)
├── .env.example                # Environment variable template
├── .gitignore
│
├── src/
│   └── congestion_pricing/
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       ├── cli.py              # Typer CLI definitions
│       │
│       ├── data/               # Data ingestion and processing
│       │   ├── __init__.py
│       │   ├── download.py     # Fetch raw data
│       │   ├── process.py      # Clean and transform
│       │   ├── validate.py     # Data quality checks
│       │   ├── schemas.py      # Pydantic/dataclass schemas
│       │   └── cities/
│       │       ├── london.py   # London-specific processing
│       │       └── nyc.py      # NYC-specific processing
│       │
│       ├── network/            # Network graph construction
│       │   ├── __init__.py
│       │   ├── graph.py        # Build NetworkX graph from OSM
│       │   ├── gtfs.py         # Parse GTFS for transit network
│       │   ├── zones.py        # Zone definitions and spatial joins
│       │   └── capacity.py     # Capacity estimation
│       │
│       ├── models/             # Model implementations
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract base class for models
│       │   ├── demand_response.py   # 3.1 Linear/nonlinear
│       │   ├── equilibrium.py       # 3.2 Network equilibrium
│       │   ├── optimization.py      # 3.3 Policy optimization
│       │   ├── game_theory.py       # 3.4 Game-theoretic
│       │   ├── abm.py               # 3.5 Agent-based
│       │   ├── stochastic.py        # 3.6 Stochastic extensions
│       │   ├── dynamic.py           # 3.7 Markov/dynamic
│       │   ├── ml.py                # 3.8 Statistical/ML
│       │   └── systems.py           # 3.9 Complexity synthesis
│       │
│       ├── evaluation/         # Validation and comparison
│       │   ├── __init__.py
│       │   ├── metrics.py      # GEH, RMSE, etc.
│       │   ├── validation.py   # Holdout and cross-validation
│       │   ├── comparison.py   # Cross-model comparison
│       │   └── dashboard.py    # Dashboard generation
│       │
│       ├── policy/             # Policy scenario definitions
│       │   ├── __init__.py
│       │   ├── scenario.py     # Scenario dataclasses
│       │   ├── factory.py      # ScenarioFactory
│       │   └── results.py      # ModelResult dataclasses
│       │
│       └── utils/              # Shared utilities
│           ├── __init__.py
│           ├── io.py           # File I/O helpers
│           ├── geo.py          # Geospatial utilities
│           ├── parallel.py     # Parallelization helpers
│           └── logging.py      # Logging configuration
│
├── configs/                    # Configuration files
│   ├── cities/
│   │   ├── london.yaml         # London-specific config
│   │   └── nyc.yaml            # NYC-specific config
│   ├── models/
│   │   ├── equilibrium.yaml    # Model hyperparameters
│   │   ├── abm.yaml
│   │   └── ml.yaml
│   └── scenarios/
│       ├── baseline.yaml
│       ├── cordon_10.yaml
│       └── time_varying.yaml
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_eda_network.ipynb    # Exploratory data analysis
│   ├── 02_eda_demand.ipynb
│   ├── 03_baseline_validation.ipynb
│   ├── 04_equilibrium_calibration.ipynb
│   ├── 05_abm_sensitivity.ipynb
│   ├── 06_ml_training.ipynb
│   ├── 07_policy_comparison.ipynb
│   ├── 08_equity_analysis.ipynb
│   └── dashboard.py            # Streamlit dashboard
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_data/              # Test data fixtures
│   │   ├── small_network.gpickle
│   │   └── sample_od.parquet
│   ├── unit/
│   │   ├── test_graph.py
│   │   ├── test_equilibrium.py
│   │   ├── test_demand_response.py
│   │   └── test_metrics.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_scenario_run.py
│
├── data/                       # Data directory (gitignored except structure)
│   ├── raw/                    # Downloaded raw data
│   │   └── .gitkeep
│   ├── processed/              # Processed data products
│   │   └── .gitkeep
│   └── results/                # Model outputs
│       └── .gitkeep
│
├── docs/                       # Documentation
│   ├── technical_design.md     # Technical design document
│   ├── policy_memo_template.md # Policy memo template
│   ├── api/                    # API reference (auto-generated)
│   └── tutorials/              # User tutorials
│
└── scripts/                    # Utility scripts
    ├── setup_data.sh           # One-time data setup
    └── run_all_scenarios.sh    # Batch scenario runner
```

### 5.2 Package Choices

#### 5.2.1 Core Dependencies

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `python` | >=3.11 | Runtime | Modern features (match, typing) |
| `numpy` | >=1.26 | Numerical | Standard for scientific computing |
| `pandas` | >=2.1 | Tabular data | Industry standard; PyArrow backend |
| `pyarrow` | >=14.0 | Parquet I/O | Fast columnar storage |
| `networkx` | >=3.2 | Graph algorithms | Flexible; good for prototyping |
| `geopandas` | >=0.14 | Geospatial | Spatial joins, geometry operations |
| `shapely` | >=2.0 | Geometry | Required by geopandas |

#### 5.2.2 Network and Data

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `osmnx` | >=1.7 | OSM download/processing | Standard for OSM network extraction |
| `gtfs-kit` | >=5.0 | GTFS parsing | Clean API for transit data |
| `requests` | >=2.31 | HTTP downloads | Standard library extension |
| `pyyaml` | >=6.0 | Config files | Human-readable configuration |

#### 5.2.3 Modeling

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `scipy` | >=1.11 | Optimization, stats | Frank-Wolfe, distributions |
| `cvxpy` | >=1.4 | Convex optimization | Toll optimization (Section 3.3) |
| `scikit-learn` | >=1.3 | ML models | Standard ML toolkit |
| `statsmodels` | >=0.14 | Econometrics | DiD, regression (Section 3.8) |
| `lightgbm` | >=4.1 | Gradient boosting | Fast, accurate tabular ML |
| `numba` | >=0.58 | JIT compilation | ABM performance (Section 3.5) |

#### 5.2.4 Visualization and CLI

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `matplotlib` | >=3.8 | Static plots | Standard; publication-quality |
| `plotly` | >=5.18 | Interactive plots | Dashboard visualizations |
| `streamlit` | >=1.28 | Dashboard app | Quick interactive dashboards |
| `typer` | >=0.9 | CLI framework | Modern, type-hint-based CLI |
| `rich` | >=13.0 | Terminal formatting | Pretty console output |

#### 5.2.5 Development

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `pytest` | >=7.4 | Testing | Standard test framework |
| `pytest-cov` | >=4.1 | Coverage | Test coverage reports |
| `ruff` | >=0.1 | Linting/formatting | Fast, replaces flake8+black |
| `mypy` | >=1.7 | Type checking | Static type verification |
| `pre-commit` | >=3.5 | Git hooks | Enforce quality on commit |

#### 5.2.6 Optional Dependencies

| Package | Purpose | When Needed |
|---------|---------|-------------|
| `python-igraph` | Faster graph algorithms | Large networks (>100k edges) |
| `dask` | Out-of-core computing | Very large datasets |
| `mlflow` | Experiment tracking | ML model versioning |
| `dvc` | Data versioning | Reproducible data pipelines |
| `pytorch` | Deep learning | LSTM/Transformer models (3.8) |

### 5.3 CLI Commands

The CLI is implemented using Typer for type-safe argument parsing.

#### 5.3.1 CLI Structure

```python
# src/congestion_pricing/cli.py

import typer
from pathlib import Path
from typing import Optional, List

app = typer.Typer(name="congestion-pricing", help="Urban congestion pricing models")
data_app = typer.Typer(help="Data operations")
model_app = typer.Typer(help="Model operations")
eval_app = typer.Typer(help="Evaluation operations")

app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")
```

#### 5.3.2 Data Commands

```bash
# Download raw data for a city
python -m congestion_pricing data fetch \
    --city london \
    --datasets osm,gtfs,traffic,census \
    --output data/raw/london/

# Verify data integrity
python -m congestion_pricing data verify \
    --city london \
    --checksums data/raw/london/checksums.sha256

# Process raw data into model-ready format
python -m congestion_pricing data process \
    --city london \
    --input data/raw/london/ \
    --output data/processed/london/ \
    --config configs/cities/london.yaml

# Generate data quality report
python -m congestion_pricing data report \
    --city london \
    --output reports/data_quality_london.html
```

#### 5.3.3 Network Commands

```bash
# Build network graph from processed data
python -m congestion_pricing data build-graph \
    --city london \
    --input data/processed/london/ \
    --output data/processed/london/network/graph.gpickle \
    --simplify \
    --add-transit

# Validate network connectivity
python -m congestion_pricing data validate-network \
    --graph data/processed/london/network/graph.gpickle
```

#### 5.3.4 Model Commands

```bash
# Run a single model on a scenario
python -m congestion_pricing model run \
    --model equilibrium \
    --scenario configs/scenarios/cordon_10.yaml \
    --output data/results/cordon_10/equilibrium/ \
    --config configs/models/equilibrium.yaml

# Run all models on a scenario
python -m congestion_pricing model run-all \
    --scenario configs/scenarios/cordon_10.yaml \
    --output data/results/cordon_10/ \
    --parallel 4

# Fit a model to data (for ML models)
python -m congestion_pricing model fit \
    --model ml \
    --data data/processed/london/ \
    --output models/ml_london.pkl \
    --config configs/models/ml.yaml

# Optimize toll schedule
python -m congestion_pricing model optimize \
    --objective welfare \
    --constraints configs/scenarios/constraints.yaml \
    --output data/results/optimal_toll.yaml
```

#### 5.3.5 Evaluation Commands

```bash
# Evaluate model against observed data
python -m congestion_pricing eval validate \
    --results data/results/cordon_10/equilibrium/ \
    --observed data/processed/london/validation/ \
    --output reports/validation_equilibrium.html

# Compare multiple models
python -m congestion_pricing eval compare \
    --results data/results/cordon_10/ \
    --output reports/comparison_cordon_10.html

# Generate full report
python -m congestion_pricing eval report \
    --results data/results/ \
    --output reports/full_report.html \
    --format html
```

#### 5.3.6 Dashboard Command

```bash
# Launch interactive dashboard
python -m congestion_pricing dashboard \
    --results data/results/ \
    --port 8501
```

### 5.4 Reproducibility Infrastructure

#### 5.4.1 Environment Management (uv)

```toml
# pyproject.toml
[project]
name = "congestion-pricing"
version = "0.1.0"
description = "Urban congestion pricing modeling framework"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "networkx>=3.2",
    "geopandas>=0.14",
    "osmnx>=1.7",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "typer>=0.9",
    "pyyaml>=6.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
    "pre-commit>=3.5",
]
ml = [
    "lightgbm>=4.1",
    "statsmodels>=0.14",
]
viz = [
    "matplotlib>=3.8",
    "plotly>=5.18",
    "streamlit>=1.28",
]
all = [
    "congestion-pricing[dev,ml,viz]",
]

[project.scripts]
congestion-pricing = "congestion_pricing.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=congestion_pricing --cov-report=term-missing"
```

#### 5.4.2 Setup Instructions

```bash
# Clone repository
git clone https://github.com/org/congestion-pricing.git
cd congestion-pricing

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -m congestion_pricing --help
pytest tests/
```

#### 5.4.3 Data Versioning (DVC - Optional)

```yaml
# dvc.yaml
stages:
  fetch_london:
    cmd: python -m congestion_pricing data fetch --city london
    deps:
      - src/congestion_pricing/data/download.py
    outs:
      - data/raw/london/

  process_london:
    cmd: python -m congestion_pricing data process --city london
    deps:
      - data/raw/london/
      - src/congestion_pricing/data/process.py
    outs:
      - data/processed/london/

  build_graph:
    cmd: python -m congestion_pricing data build-graph --city london
    deps:
      - data/processed/london/
    outs:
      - data/processed/london/network/graph.gpickle
```

```bash
# Initialize DVC
dvc init

# Track data with DVC
dvc add data/raw/london/

# Reproduce pipeline
dvc repro
```

#### 5.4.4 Experiment Tracking (MLflow - Optional)

```python
# src/congestion_pricing/utils/tracking.py

import mlflow
from pathlib import Path

def setup_mlflow(experiment_name: str = "congestion_pricing"):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

def log_model_run(model_name: str, scenario: dict, results: dict):
    """Log a model run to MLflow."""
    with mlflow.start_run(run_name=f"{model_name}_{scenario['name']}"):
        # Log parameters
        mlflow.log_params({
            "model": model_name,
            "scenario": scenario["name"],
            "toll_level": scenario.get("toll", {}).get("rates", {}),
        })

        # Log metrics
        mlflow.log_metrics({
            "mean_travel_time": results["aggregates"]["mean_travel_time"],
            "mode_share_transit": results["aggregates"]["mode_share_transit"],
            "revenue": results["aggregates"]["toll_revenue"],
        })

        # Log artifacts
        mlflow.log_artifact(results["output_path"])
```

#### 5.4.5 CI/CD Configuration

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v2

      - name: Install dependencies
        run: |
          uv venv
          uv pip install -e ".[dev]"

      - name: Lint
        run: |
          source .venv/bin/activate
          ruff check src/ tests/

      - name: Type check
        run: |
          source .venv/bin/activate
          mypy src/

      - name: Test
        run: |
          source .venv/bin/activate
          pytest tests/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
```

---

## 6. Milestones and Sequencing

This section defines a staged implementation plan with concrete outputs and acceptance criteria for each stage.

### 6.0 Stage 0: Data Ingestion and Baseline EDA

**Objective:** Establish data pipeline and understand data characteristics.

#### 6.0.1 Tasks

1. Set up repository structure and development environment
2. Implement data download scripts for primary city (London or NYC)
3. Implement data processing pipeline
4. Generate exploratory data analysis notebooks
5. Document data quality issues and limitations

#### 6.0.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw data | `data/raw/{city}/` | Downloaded OSM, GTFS, traffic, census |
| Processed data | `data/processed/{city}/` | Parquet files per schema |
| EDA notebook | `notebooks/01_eda_network.ipynb` | Network statistics, maps |
| EDA notebook | `notebooks/02_eda_demand.ipynb` | Demand patterns, temporal profiles |
| Data report | `reports/data_quality.html` | Quality metrics, missing data |

#### 6.0.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Data completeness | % of required fields non-null | > 95% |
| Network connectivity | Largest connected component | > 99% of nodes |
| Demand coverage | % of zones with demand data | > 90% |
| Reproducibility | Pipeline runs from scratch | Pass |
| Documentation | Data dictionary complete | Yes |

---

### 6.1 Stage 1: Network Graph and Baseline Speed/Flow Validation

**Objective:** Build validated network model matching observed conditions.

#### 6.1.1 Tasks

1. Build NetworkX graph from OSM with capacity estimates
2. Integrate GTFS for transit network
3. Define zone system and centroids
4. Match observed traffic counts to network links
5. Validate network against observed speeds and counts

#### 6.1.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Network graph | `data/processed/{city}/network/graph.gpickle` | Directed graph with attributes |
| Zone definitions | `data/processed/{city}/zones/zones.geojson` | Zone polygons with attributes |
| Validation notebook | `notebooks/03_baseline_validation.ipynb` | Flow/speed comparison |
| Calibration report | `reports/network_calibration.html` | GEH statistics, maps |

#### 6.1.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Network completeness | All major roads included | Visual inspection pass |
| Connectivity | OD pairs connected | > 99% |
| Free-flow times | Correlation with observed | R > 0.9 |
| Link matching | Observed counts matched to links | > 80% |
| Zone coverage | All zones have centroid connectors | 100% |

---

### 6.2 Stage 2: Static Equilibrium Model and Simple Toll Counterfactual

**Objective:** Implement and validate static traffic assignment; run first policy scenarios.

#### 6.2.1 Tasks

1. Implement Wardrop equilibrium solver (Frank-Wolfe)
2. Calibrate BPR parameters from speed-flow data
3. Validate equilibrium flows against observed counts
4. Run baseline (no-toll) scenario
5. Run simple cordon toll scenarios (e.g., $5, $10, $15)
6. Compare results to London 2003 data (if available)

#### 6.2.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Equilibrium model | `src/congestion_pricing/models/equilibrium.py` | Tested implementation |
| Calibration notebook | `notebooks/04_equilibrium_calibration.ipynb` | BPR fitting |
| Baseline results | `data/results/baseline/equilibrium/` | Flows, times |
| Toll scenario results | `data/results/cordon_{x}/equilibrium/` | Multiple toll levels |
| Validation report | `reports/equilibrium_validation.html` | GEH, counterfactual check |

#### 6.2.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Convergence | Relative gap | < 0.01 |
| Flow validation | GEH < 5 | > 85% of links |
| Travel time validation | MAPE | < 15% |
| Counterfactual direction | Traffic reduction with toll | Correct sign |
| Runtime | Full assignment | < 10 min |

---

### 6.3 Stage 3: Elasticity Estimation and Discrete Choice

**Objective:** Estimate behavioral parameters from data; implement demand response model.

#### 6.3.1 Tasks

1. Implement demand response model (Section 3.1)
2. Estimate price elasticity from historical toll changes (if data exists)
3. Estimate mode choice model from census/survey data
4. Calibrate VOT distribution from income data
5. Validate against observed mode shares

#### 6.3.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Demand model | `src/congestion_pricing/models/demand_response.py` | Implementation |
| Elasticity estimates | `configs/models/elasticities.yaml` | Calibrated parameters |
| Mode choice notebook | `notebooks/05_mode_choice.ipynb` | Estimation results |
| Validation report | `reports/demand_validation.html` | Mode share comparison |

#### 6.3.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Elasticity sign | Negative for own-price | Yes |
| Elasticity magnitude | Within literature range | -0.1 to -0.5 |
| Mode share validation | Absolute error | < 3 pp |
| Model convergence | Logit estimation | Converged |
| Cross-validation | Out-of-sample R2 | > 0.7 |

---

### 6.4 Stage 4: Toll Schedule Optimization

**Objective:** Implement optimization model to find welfare-maximizing tolls.

#### 6.4.1 Tasks

1. Implement optimization model (Section 3.3)
2. Define objective function (welfare, revenue, equity variants)
3. Implement constraints (revenue floor, equity bound)
4. Solve for optimal time-varying toll
5. Sensitivity analysis on constraint parameters

#### 6.4.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Optimization model | `src/congestion_pricing/models/optimization.py` | Implementation |
| Optimal toll | `data/results/optimal_toll/` | Schedule and outcomes |
| Sensitivity notebook | `notebooks/06_optimization.ipynb` | Pareto frontier |
| Policy brief | `reports/optimal_toll_memo.md` | Recommendations |

#### 6.4.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Constraint satisfaction | All constraints met | Yes |
| Improvement over baseline | Welfare gain | > 0 |
| Solver convergence | Optimality gap | < 1% |
| Sensitivity | Results stable to perturbations | Yes |
| Computation time | Full optimization | < 1 hour |

---

### 6.5 Stage 5: Agent-Based Model and Dynamic Adaptation

**Objective:** Implement ABM for heterogeneous agents and day-to-day dynamics.

#### 6.5.1 Tasks

1. Implement agent-based model (Section 3.5)
2. Synthesize agent population from census
3. Implement learning and habit rules
4. Simulate day-to-day adaptation after toll introduction
5. Implement Markov/dynamic model (Section 3.7)
6. Compare steady states to equilibrium model

#### 6.5.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| ABM model | `src/congestion_pricing/models/abm.py` | Implementation |
| Dynamic model | `src/congestion_pricing/models/dynamic.py` | Implementation |
| Population | `data/processed/{city}/agents/population.parquet` | Synthetic agents |
| Trajectory results | `data/results/cordon_10/abm/trajectory.parquet` | Daily metrics |
| Dynamics notebook | `notebooks/07_abm_dynamics.ipynb` | Adaptation curves |

#### 6.5.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Steady-state match | ABM vs equilibrium | CV < 10% |
| Adaptation time | Days to 95% of equilibrium | 20-100 days |
| Heterogeneity | Variance in agent outcomes | Positive |
| Mode share validation | Steady-state vs observed | Within 3 pp |
| Performance | 100-day simulation | < 30 min |

---

### 6.6 Stage 6: Stochastic Reliability and Robustness

**Objective:** Quantify uncertainty and stress-test policy conclusions.

#### 6.6.1 Tasks

1. Implement stochastic extensions (Section 3.6)
2. Estimate incident/weather shock parameters
3. Run Monte Carlo simulations for reliability metrics
4. Implement game-theoretic extensions (Section 3.4) if ride-hail modeled
5. Robustness analysis across parameter uncertainty
6. Document which conclusions are robust vs. sensitive

#### 6.6.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Stochastic model | `src/congestion_pricing/models/stochastic.py` | Implementation |
| Monte Carlo results | `data/results/cordon_10/stochastic/` | Distributions |
| Reliability notebook | `notebooks/08_reliability.ipynb` | Buffer time analysis |
| Robustness report | `reports/robustness_analysis.html` | Sensitivity summary |

#### 6.6.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Monte Carlo convergence | SE of mean < target | < 5% of mean |
| Reliability calibration | Buffer time vs observed | Within 20% |
| Robustness documentation | Parameters tested | All key params |
| Policy conclusion stability | Sign of effect stable | > 80% of draws |

---

### 6.7 Stage 7: Causal Evaluation (If Natural Experiment Data Exists)

**Objective:** Estimate causal effects of actual policy changes.

#### 6.7.1 Tasks

1. Implement ML predictive models (Section 3.8)
2. If before/after data exists: implement DiD analysis
3. If cross-city data exists: implement synthetic control
4. Validate causal design (parallel trends, placebo tests)
5. Compare causal estimates to model predictions

#### 6.7.2 Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| ML models | `src/congestion_pricing/models/ml.py` | Implementation |
| Causal analysis | `notebooks/09_causal_analysis.ipynb` | DiD/synthetic control |
| Prediction models | `models/ml_{city}.pkl` | Trained models |
| Causal report | `reports/causal_evaluation.html` | Effect estimates |

#### 6.7.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Parallel trends | Pre-trend coefficient | Not significant |
| Placebo tests | Fake treatment effects | Not significant |
| Prediction accuracy | Out-of-sample RMSE | < baseline |
| Causal vs model | Predicted effect vs model | Same order of magnitude |

---

### 6.8 Summary: Stage Dependencies

```
Stage 0: Data Ingestion
    |
    v
Stage 1: Network Validation
    |
    v
Stage 2: Equilibrium Model  -----> Stage 4: Optimization
    |                                  |
    v                                  v
Stage 3: Demand Estimation         (uses equilibrium)
    |
    +---> Stage 5: ABM/Dynamic
    |         |
    |         v
    +---> Stage 6: Stochastic
    |
    v
Stage 7: Causal Evaluation (if data available)
```

**Critical path:** Stages 0-2 are sequential prerequisites. Stages 3-6 can proceed in parallel after Stage 2. Stage 7 depends on data availability.

---

## 7. Writeup Templates

### 7.1 Technical Design Document Outline

```markdown
# Congestion Pricing Modeling Framework: Technical Design

## 1. Executive Summary
- Purpose of the framework
- Key capabilities
- Target users

## 2. Architecture Overview
- System diagram
- Module relationships
- Data flow

## 3. Data Layer
### 3.1 Data Sources
- Description of each source
- Update frequency
- Licensing

### 3.2 Data Pipeline
- Ingestion process
- Transformation steps
- Quality checks

### 3.3 Data Schema
- Table definitions
- Relationships
- Versioning strategy

## 4. Model Layer
### 4.1 Model Inventory
- List of implemented models
- Purpose of each
- Computational requirements

### 4.2 Common Interface
- Scenario specification
- Output format
- Error handling

### 4.3 Model-Specific Details
- For each model:
  - Mathematical formulation
  - Algorithm description
  - Parameters and calibration
  - Limitations

## 5. Evaluation Layer
### 5.1 Validation Framework
- Holdout design
- Metrics definitions
- Acceptance criteria

### 5.2 Cross-Model Comparison
- Comparison methodology
- Consistency checks
- Disagreement resolution

## 6. Infrastructure
### 6.1 Environment Setup
- Dependencies
- Installation
- Configuration

### 6.2 CLI Reference
- Command documentation
- Examples
- Troubleshooting

### 6.3 Testing Strategy
- Unit tests
- Integration tests
- Regression tests

## 7. Extension Points
- Adding new cities
- Adding new models
- Custom analysis

## Appendices
- A. Glossary
- B. API Reference
- C. Configuration Reference
```

### 7.2 Policy Memo Template

```markdown
# Policy Memo: [Congestion Pricing Scenario Name]

**Date:** [YYYY-MM-DD]
**Prepared for:** [Audience]
**Prepared by:** [Authors]

---

## Executive Summary

[2-3 paragraph summary of key findings, suitable for senior decision-makers]

**Key Finding 1:** [One sentence]
**Key Finding 2:** [One sentence]
**Key Finding 3:** [One sentence]

---

## 1. Policy Question

[What decision is this analysis informing?]

## 2. Scenarios Analyzed

| Scenario | Description | Toll Structure |
|----------|-------------|----------------|
| Baseline | Current conditions | None |
| Scenario A | [Description] | [Details] |
| Scenario B | [Description] | [Details] |

## 3. Methodology

### 3.1 Models Used
- [List models applied]
- [Rationale for selection]

### 3.2 Data Sources
- [Key data inputs]
- [Time period covered]

### 3.3 Key Assumptions
| Assumption | Value | Justification |
|------------|-------|---------------|
| Value of time | $X/hr | [Source] |
| Price elasticity | -X.X | [Source] |
| [Other] | [Value] | [Source] |

## 4. Results

### 4.1 Traffic Impacts

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| Cordon entries/day | X | X (-Y%) | X (-Z%) |
| Mean travel time | X min | X min | X min |
| Transit mode share | X% | X% | X% |

### 4.2 Financial Impacts

| Metric | Scenario A | Scenario B |
|--------|------------|------------|
| Annual revenue | $X M | $X M |
| Collection costs | $X M | $X M |
| Net revenue | $X M | $X M |

### 4.3 Equity Impacts

| Income Quintile | Scenario A Impact | Scenario B Impact |
|-----------------|-------------------|-------------------|
| Q1 (lowest) | -$X/year | -$X/year |
| Q2 | -$X/year | -$X/year |
| Q3 | +$X/year | +$X/year |
| Q4 | +$X/year | +$X/year |
| Q5 (highest) | -$X/year | -$X/year |

### 4.4 Environmental Impacts

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| CO2 (tons/year) | X | X (-Y%) | X (-Z%) |
| VKT (M km/year) | X | X | X |

## 5. Uncertainty and Limitations

### 5.1 Model Agreement
[Where do models agree/disagree?]

### 5.2 Key Uncertainties
| Uncertainty | Impact on Conclusions |
|-------------|----------------------|
| Demand elasticity | [High/Medium/Low] |
| Induced demand | [High/Medium/Low] |
| [Other] | [Assessment] |

### 5.3 Data Limitations
[What data gaps affect confidence?]

## 6. Recommendations

1. **[Recommendation 1]:** [Brief description]
   - Rationale: [Why]
   - Caveats: [Conditions]

2. **[Recommendation 2]:** [Brief description]
   - Rationale: [Why]
   - Caveats: [Conditions]

## 7. Next Steps

- [ ] [Action item 1]
- [ ] [Action item 2]
- [ ] [Action item 3]

---

## Appendix A: Detailed Results

[Tables and figures with full results]

## Appendix B: Model Validation Summary

[Goodness-of-fit metrics]

## Appendix C: Sensitivity Analysis

[How results change with different assumptions]
```

### 7.3 Model Documentation Template

For each model module, include:

```markdown
# Model: [Model Name]

## Overview

**Purpose:** [Predict/Explain/Prescribe/Explore]
**Complexity:** [Low/Medium/High]
**Typical runtime:** [Seconds/Minutes/Hours]

## Mathematical Formulation

[LaTeX or plaintext math describing the model]

## Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| network | nx.DiGraph | Yes | Road network graph |
| demand | DataFrame | Yes | OD demand matrix |
| toll | TollSchedule | No | Toll specification |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| flows | DataFrame | Link flows by period |
| times | DataFrame | Travel times by OD |
| aggregates | AggregateResults | Summary metrics |

## Parameters

| Parameter | Default | Range | Calibration Method |
|-----------|---------|-------|-------------------|
| alpha | 0.15 | 0.1-0.3 | Regression on speed-flow |
| beta | 4.0 | 2-6 | Regression on speed-flow |

## Usage Example

```python
from congestion_pricing.models import EquilibriumModel
from congestion_pricing.policy import ScenarioFactory

scenario = ScenarioFactory.baseline("london")
model = EquilibriumModel(config_path="configs/models/equilibrium.yaml")
result = model.run(scenario)

print(f"Mean travel time: {result.aggregates.mean_travel_time:.1f} min")
```

## Validation

| Metric | Target | Typical Value |
|--------|--------|---------------|
| GEH < 5 | > 85% | 87% |
| TT MAPE | < 15% | 12% |

## Limitations

- [Limitation 1]
- [Limitation 2]

## References

- [Citation 1]
- [Citation 2]
```

---

## 8. Conclusion

This engineering plan provides a comprehensive roadmap for building a modular urban congestion pricing modeling framework. The plan covers:

1. **Problem formalization** with explicit decision variables, constraints, and metrics
2. **Real-world data sources** from London and NYC with documented access
3. **Nine model families** spanning traditional and modern approaches
4. **Common interfaces** enabling cross-model comparison
5. **Production-ready engineering** with testing, CI/CD, and reproducibility
6. **Staged implementation** with clear milestones and acceptance criteria
7. **Documentation templates** for technical and policy communication

The staged approach allows incremental validation while building toward a full multi-model framework. Early stages establish data and validation infrastructure that subsequent stages depend on. The common interface design ensures that new models can be added without disrupting existing functionality.

---

*Document version: 1.0*
*Last updated: [Date]*
