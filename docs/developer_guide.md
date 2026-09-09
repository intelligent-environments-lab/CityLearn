# Developer Guide

This page collects local development commands, validation checks, performance smoke tests and the current internal architecture map.

Portuguese version: [pt/developer_guide.md](pt/developer_guide.md).

## Local Environment

Use the repository virtual environment when available:

```console
.venv/bin/pip install -e .
```

Optional dependencies:

| Feature | Command |
|---|---|
| Parquet datasets | `.venv/bin/pip install pyarrow` |
| PV autosizing | `.venv/bin/pip install ".[pysam]"` |
| Build checks | `.venv/bin/pip install build twine` |

## Core Validation

| Check | Command |
|---|---|
| Full test suite | `.venv/bin/pytest -q` |
| Critical lint | `.venv/bin/python -m ruff check citylearn tests scripts/manual scripts/ci --select E9,F821` |
| Entity contract audit | `.venv/bin/python scripts/audit/audit_entity_contract.py --strict` |
| Physics audit | `.venv/bin/python scripts/audit/audit_physics.py` |
| Diff whitespace check | `git diff --check` |

The physics audit intentionally exercises resolution conversion paths, so unit-conversion warnings can be expected when the audit fixture forces dataset/schema mismatches.

## Manual Simulation Scripts

Manual utility scripts live in `scripts/manual` and are excluded from default pytest collection.

```console
.venv/bin/python scripts/manual/demo_ev_rbc.py
.venv/bin/python scripts/manual/demo_ev_rbc_export_end.py
```

Runtime benchmark:

```console
.venv/bin/python scripts/manual/bench_runtime.py --seconds 5 60 --render-modes none end --episode-steps 1200
```

CI performance smoke check:

```console
.venv/bin/python scripts/ci/perf_smoke.py --episode-steps 600 --seconds 60 --baseline-file scripts/ci/perf_baseline.json
```

Step/component profiling:

```console
.venv/bin/python scripts/audit/profile_step_breakdown.py --episode-steps 300 --seconds 60 --agent rbc --interface flat --render-mode none --no-write --table-limit 14
```

Lifecycle and KPI/BAU profiling:

```console
.venv/bin/python scripts/audit/profile_lifecycle.py --episode-steps 300 --seconds 60 --skip-exports --no-write
```

Drop `--skip-exports` when the CSV export paths themselves need to be timed.

For ad hoc code, pass `debug_timing=True` to `CityLearnEnv` and inspect `info` keys such as `apply_actions_time`, `reward_observations_time`, `next_observations_time`, `terminal_export_time` and `step_total_time`.

## Internal Architecture

Public APIs remain centered on `CityLearnEnv` and `Building`. Internal orchestration is split into service modules under `citylearn/internal`.

| Module | Responsibility |
|---|---|
| `loading.py` | Schema-driven loading and building assembly. |
| `runtime.py` | Episode runtime orchestration, `step`, action parsing, time progression and EV/charger association. |
| `building_ops.py` | Building observation/action orchestration. |
| `kpi.py` | KPI/evaluation pipeline. |
| `entity_interface.py` | Entity tables, edges, action specs and observation bundles. |

## Development Rules

| Rule | Reason |
|---|---|
| Keep schema/API changes documented in the same change. | Algorithms and datasets depend on explicit contracts. |
| Add tests for new physics or observations. | Sub-hourly and entity behavior are easy to regress silently. |
| Prefer additive observations in patch releases. | Avoid breaking trained algorithms and wrappers. |
| Run audits before tagging. | They catch contract drift outside normal unit tests. |
