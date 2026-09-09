# CityLearn 3.0.0 release notes

CityLearn 3.0.0 promotes the complete Soft-CPS Simulator development line into
the official CityLearn repository. It establishes the fork's runtime, datasets,
documentation, audits and regression tests as the official Version 3 baseline.
The imported source snapshot is `Soft-CPS-Research-Group/Simulator@3b6e3ae4`.

## Highlights

- Entity-based observations and actions alongside the classic flat Gymnasium interface.
- Dynamic building and asset topology with repeatable episode resets.
- Multi-community orchestration and portfolio KPIs.
- Sub-hourly simulation, CSV/Parquet datasets and explicit physical-unit contracts.
- Extended EV, charger, BESS, PV, deferrable appliance and escalator models.
- Single- and three-phase electrical-service constraints.
- Community-market settlement, KPI v2 and stable UI/export contracts.
- Dataset-driven robustness events, forecasts and annual REC benchmark suites.
- Expanded unit, integration, physics, contract and performance validation.

## Package migration

The official package remains `citylearn`, and Python imports remain unchanged:

```console
pip install citylearn==3.0.0
```

```python
from citylearn.citylearn import CityLearnEnv
```

Workloads that previously installed the fork distribution should replace
`softcpsrecsimulator` with `citylearn==3.0.0`.

## Compatibility

This is a major release. The project retains the established `CityLearnEnv` and
Gymnasium entry points, while adding schema fields, observation/action contracts,
KPIs and export data. Validate custom schemas, wrappers and downstream KPI/export
consumers before production upgrades.

## Validation

The release candidate passed the following checks on 2026-09-09:

- `python -m pytest -q --ignore=scripts/manual`: 458 passed, 18 warnings.
- `python scripts/audit/audit_entity_contract.py --strict`: pass.
- `python scripts/audit/audit_physics.py`: 16/16 scenarios passed.
- `python -m build`: pass; generated `citylearn-3.0.0` wheel and source archive.
- `python -m twine check dist/*`: pass for both artifacts.

The validation run exposed and fixed an audit-boundary defect: after a dynamic
topology event, historical physics checks now filter the current building view
using the membership active at the inspected timestep.
