# CityLearn

CityLearn is an open-source Farama Foundation Gymnasium environment for building energy coordination, demand response and multi-agent reinforcement learning. Version 3 integrates the extended simulator developed by the Soft-CPS Research Group, including EVs, normalized deferrable appliances, sub-hourly physics, entity observations, dataset-driven demand response, robustness events, dynamic topology, three-phase electrical service, community KPIs and performance work for large datasets.

Package name:

```console
pip install citylearn
```

Python import path:

```python
from citylearn.citylearn import CityLearnEnv
```

Default documentation is in English. Portuguese documentation is available under [docs/pt](docs/pt/).

Current source release: `3.0.2`.

## Start Here

| Link | Portuguese | Use it for |
|---|---|---|
| [Release history](docs/releases.md) | [PT](docs/pt/releases.md) | Version history, release owner, validation status and compatibility notes. |
| [How to run simulations](docs/running_simulations.md) | [PT](docs/pt/running_simulations.md) | Installation, quickstarts, CLI, `CityLearnEnv` parameters, render/export and validation. |
| [Schema reference](docs/schema_reference.md) | [PT](docs/pt/schema_reference.md) | Full schema contract: buildings, devices, PV modes, EVs, chargers, deferrables, topology and market. |
| [Dataset reference](docs/dataset_reference.md) | [PT](docs/pt/dataset_reference.md) | Required files/columns, CSV/Parquet, 15s datasets and real-data conversion. |
| [Observations reference](docs/observations_reference.md) | [PT](docs/pt/observations_reference.md) | Observation names, units, bundles, sentinels, entity tables and edges. |
| [Actions reference](docs/actions_reference.md) | [PT](docs/pt/actions_reference.md) | Flat/entity actions, ranges and physical meaning. |
| [Flat and entity interfaces](docs/interfaces_flat_entity.md) | [PT](docs/pt/interfaces_flat_entity.md) | Vector mode, entity-table mode and dynamic topology semantics. |
| [KPIs reference](docs/kpis_reference.md) | [PT](docs/pt/kpis_reference.md) | `evaluate()`, `evaluate_v2()`, KPI units and KPI families. |
| [Multi-community reference](docs/multi_community_reference.md) | [PT](docs/pt/multi_community_reference.md) | Orchestrating multiple synchronized communities and portfolio KPIs. |
| [Robustness reference](docs/robustness_reference.md) | [PT](docs/pt/robustness_reference.md) | Dataset-driven observation, forecast, action and asset availability perturbations. |
| [Data unit contract](docs/data_unit_contract.md) | [PT](docs/pt/data_unit_contract.md) | Formal contract for `kWh/step`, `kW`, prices, emissions and timesteps. |
| [Simulator features](docs/features.md) | [PT](docs/pt/features.md) | Capability inventory, including less obvious features. |
| [Developer guide](docs/developer_guide.md) | [PT](docs/pt/developer_guide.md) | Tests, audits, performance checks and internal architecture. |
| [Publishing guide](docs/publishing.md) | [PT](docs/pt/publishing.md) | PyPI release workflow and local build checks. |

Additional reference: [KPI v2 naming tree](docs/KPI_V2_TREE.md).

## Capability Snapshot

| Area | Supported |
|---|---|
| Time resolution | Hourly and sub-hourly, including 15min, 5min, 1min and 15s fixtures. |
| Dataset formats | CSV and Parquet with equivalent schema columns. |
| Real data | Power data can be converted to `kWh/step`; PV supports absolute measured generation. |
| PV | `per_kw` normalized profile mode and `absolute` measured-energy mode. |
| EVs | Charger-centric schedules, connected/incoming EVs, SOC requirements and V2G-capable actions. |
| Deferrables | Normalized cycle catalog plus sparse flexibility schedule. |
| Escalators | Aggregate standby/slow/normal control with passenger-service signals and KPIs. |
| Interfaces | Flat Gymnasium vectors and entity tables/edges for offline RL, GraphRL and Transformers. |
| Entity RL observations | Forecast bundles, physical deadline pressure, feasible action capacity and requested/limited/applied action feedback. |
| Dynamic topology | Add/remove buildings and assets during simulation in entity mode, with clean structural replay after each episode reset. |
| Demand response | Dataset-driven DSO/TSO flexibility requests in entity observations, with settlement and KPIs. |
| Multi-community | Synchronized orchestration of multiple independent communities with portfolio KPI rows. |
| Robustness | Optional dataset-driven observation, forecast, action-channel and logical asset outages with KPIs. |
| Three phase | Phase connections, headroom, phase power, violations and phase KPIs. |
| Community market | Local settlement, import weights, savings and self-consumption KPIs. |
| Performance | Windowed loading, shared weather/pricing/carbon cache, runtime profiling and Parquet for large 15s datasets. |
| Validation | Unit tests, golden KPI tests, physics audit and strict entity contract audit. |

## Quickstart

```python
import numpy as np
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
    interface="flat",
    episode_time_steps=24,
    render_mode="none",
)

observations, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    actions = [np.zeros(space.shape, dtype="float32") for space in env.action_space]
    observations, reward, terminated, truncated, info = env.step(actions)

kpis = env.evaluate_v2()
```

Entity interface:

```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json",
    interface="entity",
    topology_mode="dynamic",
)

observations, info = env.reset()
specs = env.entity_specs
```

Compact 15s parquet dataset with dynamic asset changes:

```python
env = CityLearnEnv(
    "data/datasets/citylearn_three_phase_dynamic_asset_changes_demo_15s_parquet/schema.json",
    interface="entity",
    topology_mode="dynamic",
)
```

Demand response dataset:

```python
env = CityLearnEnv(
    "data/datasets/citylearn_challenge_2022_phase_all_demand_response/schema.json",
    interface="entity",
)
```

Robustness dataset:

```python
env = CityLearnEnv(
    "data/datasets/citylearn_challenge_2022_phase_all_robustness/schema.json",
    interface="entity",
)
```

Multi-community portfolio:

```python
from citylearn.multi_community import MultiCommunityEnv

env = MultiCommunityEnv(
    communities=[
        {
            "community_id": "community_a",
            "schema": "data/datasets/community_a/schema.json",
            "env_kwargs": {"interface": "entity", "episode_time_steps": 48},
            "weight": 1.0,
        },
        {
            "community_id": "community_b",
            "schema": "data/datasets/community_b/schema.json",
            "env_kwargs": {"interface": "entity", "episode_time_steps": 48},
            "weight": 1.0,
        },
    ],
)
```

## Unit Contract

| Quantity | Unit |
|---|---|
| Dataset energy columns | `kWh/step` |
| PV `generation_mode="absolute"` | `kWh/step` |
| PV `generation_mode="per_kw"` | `W/kW` profile multiplied by installed power |
| Power limits and ratings | `kW` |
| Prices | currency/kWh |
| Carbon intensity | kgCO2/kWh |
| Deferrable cycle `load_profile` | `kWh/step` |

Real power data conversion:

```text
kWh_per_step = kW * seconds_per_time_step / 3600
```

## Validation

Recommended pre-release checks:

```console
.venv/bin/pytest -q
.venv/bin/python scripts/audit/audit_entity_contract.py --strict
.venv/bin/python scripts/audit/audit_physics.py
```

See [Developer guide](docs/developer_guide.md) for lint, smoke simulations, benchmarks and architecture notes.

## Documentation and UI

The CityLearn documentation covers the base concepts, examples and Version 3 extensions: [official docs](https://www.citylearn.net/).

CityLearn UI is a visual dashboard for inspecting simulation data and KPIs:

| Resource | Link |
|---|---|
| Hosted web app | <https://citylearnui.netlify.app/> |
| Open-source UI | <https://github.com/Soft-CPS-Research-Group/citylearn-ui> |

The KPI export consumed by the UI is generated from `evaluate_v2()` by default, while `evaluate()` remains available for legacy workflows.
