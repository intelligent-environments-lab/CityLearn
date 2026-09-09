# Simulator Features

This page summarizes the simulator capabilities, including features that are easy to miss when reading only the README.

Portuguese version: [pt/features.md](pt/features.md).

## Core Simulation

| Feature | What it provides |
|---|---|
| Gymnasium environment | Standard `reset/step`, action spaces and observation spaces. |
| Multi-building district | Community-level simulation with multiple buildings and DERs. |
| Multi-agent or central-agent | One vector per building or one centralized controller. |
| HVAC and thermal storage | Cooling, heating and domestic hot water with storage. |
| BESS | Electrical storage with SOC, efficiency, power curves and degradation. |
| PV | Generation in normalized per-kW mode or absolute measured-energy mode. |
| EVs and chargers | Charger-centric schedules, connected/incoming EVs, V2G-capable action path. |
| Deferrable appliances | Normalized cycles, flexibility windows and service KPIs. |
| Demand response | Dataset-driven DSO/TSO flexibility requests with entity observations, settlement and KPIs. |
| Multi-community orchestration | Runs multiple synchronized `CityLearnEnv` communities and reports portfolio KPIs. |
| Robustness events | Optional dataset-driven degradation of observations, forecasts, actions and logical asset availability. |

## Physics and Units

| Feature | Why it matters |
|---|---|
| Sub-hourly support | Tested support for 1h, 15min, 5min, 1min and 15s scenarios. |
| Explicit unit contract | Dataset energy is `kWh/step`; power limits are `kW`. |
| Step-aware action conversion | Power actions are converted to step energy using `seconds_per_time_step`. |
| PV absolute mode | Real measured PV generation can be used directly. |
| Physics invariant checks | Optional runtime physical consistency checks. |
| Audit scripts | `audit_physics.py` and `audit_entity_contract.py --strict`. |

## Entity Interface

| Feature | Why it matters |
|---|---|
| Entity tables | `district`, `building`, `charger`, `ev`, `storage`, `pv`, `deferrable_appliance`. |
| Relational edges | Ready for GraphRL/GNN models. |
| Stable IDs | Enables memory per entity and dynamic topology. |
| Observation bundles | Controls the richness and cost of observations. |
| Demand response bundle | Adds active request, baseline, prices and previous delivery/shortfall to the district table. |
| Robustness bundle | Adds active robustness state and previous-step corruption counters to the district table. |
| Machine-readable specs | `env.entity_specs` exposes IDs, features, units and bundles. |
| Observation copies | Prevents agents from mutating internal simulator state by accident. |

## Dynamic Topology

| Feature | What it does |
|---|---|
| `add_member` / `remove_member` | Buildings can enter or leave the community. |
| `add_asset` / `remove_asset` | Chargers, PV, BESS and deferrables can change mid-episode. |
| Space/spec refresh | Action spaces, observation spaces and entity specs follow topology changes. |
| Lifecycle metadata | `born_at`, `removed_at`, `active`. |
| Safe deferrable removal | Cancels pending/running cycles and clears future consumption. |

## Three-Phase Electrical Service

| Feature | What it measures or controls |
|---|---|
| Phase connection | Chargers and storage can use `L1`, `L2`, `L3` or `all_phases`. |
| Headroom observations | Import/export headroom by building and phase. |
| Phase power observations | Current phase power in the entity core bundle. |
| Violation KPIs | Total violation energy and event count. |
| Phase imbalance KPI | Average imbalance across phases. |
| Phase peak KPIs | Import/export peaks for L1/L2/L3. |

## Community Market and KPIs

| Feature | What it provides |
|---|---|
| Local settlement | Local matching between exporters and importers. |
| Weighted imports | Import allocation by member weights. |
| Market savings | Settled cost compared with counterfactual grid settlement. |
| Community self-consumption | Local import share and traded energy KPIs. |
| KPI v2 naming tree | Structured KPI names by level/family/subfamily/metric/unit. |
| Golden tests | Hand-calculated KPI tests for critical metrics. |

## Multi-Community

| Feature | What it provides |
|---|---|
| `MultiCommunityEnv` | Public wrapper in `citylearn.multi_community` for multiple independent communities. |
| Synchronized stepping | Every child community must share `seconds_per_time_step`, episode length, interface and `central_agent` mode. |
| Independent DR | Each community keeps its own demand-response requests, observations, settlement and KPIs. |
| Portfolio KPI rows | `evaluate_v2()` adds `level="portfolio"` rows with summed totals and weighted ratios. |
| Export layout | Child KPIs go to per-community subfolders, with a global `exported_kpis_multi_community.csv`. |

## Robustness

| Feature | What it provides |
|---|---|
| Dataset events | CSV/Parquet event files configure deterministic perturbations. |
| Modular switches | Observations, forecasts, actions and assets can be enabled independently. |
| Real-state separation | Observation/forecast corruption does not change rewards, DR settlement or physical KPIs. |
| Imperfect action channel | Dropout, noise, bias, stuck, delay and clip can change the applied control action. |
| Logical asset outage | Assets stay in the topology, while telemetry and/or control can become unavailable. |
| Robustness KPIs | Counts event coverage, corruptions, missing observations and action dropouts. |

## Dataset and Performance

| Feature | Benefit |
|---|---|
| CSV and Parquet | Same data contract with efficient storage options. |
| Windowed loader | Loads only the requested simulation window. |
| Shared time-series cache | Reuses weather/pricing/carbon across buildings. |
| 15s fixtures | Tests small energy values and rounding behavior. |
| Render modes | `none`, `during`, `end` balance training speed and debugging output. |

## UI and Export

| Feature | What it enables |
|---|---|
| Render/export CSV | Post-episode inspection. |
| KPI export | Dashboard and worker integration. |
| CityLearn UI compatibility | Consumption of `evaluate_v2()` outputs. |
| Start date | Human-readable timestamps in outputs. |

## Testing and Production Readiness

| Area | Coverage |
|---|---|
| Sub-hourly | 1min, 5min, 15min, 1h and 15s. |
| KPIs | Golden and sanity tests. |
| Entity contract | Strict audit and snapshots. |
| Dynamic topology | Add/remove members/assets and action-space refresh. |
| Deferrables | Parser, actions, KPIs and topology add/remove. |
| Market/phases | Conservation and coherence checks for import/export/phases. |
