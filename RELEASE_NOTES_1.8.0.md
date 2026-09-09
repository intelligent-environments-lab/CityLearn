# softcpsrecsimulator 1.8.0 release notes

This release introduces the canonical annual REC benchmark suite and the
Simulator contracts required to run it reproducibly. It is a minor release:
existing flat and entity scenarios remain supported, while causal forecasts,
terminal-boundary handling and the new benchmark schemas are additive.

## Canonical annual REC benchmark suite

The repository now contains four deterministic hybrid families covering the
full 2023 calendar year at 15-minute resolution:

| Family | Members | PV | BESS | Chargers | Deferrables | Variants |
|---|---:|---:|---:|---:|---:|---|
| `MICRO-4-Q` | 4 | 2 | 1 | 2 | 1 | Micro |
| `CORE-15-STRIPPED` | 15 | 7 | 3 | 7 | 4 | Stripped |
| `CORE-30` | 30 | 15 | 6 | 15 | 8 | Nominal, Safety, Health, Dynamic, Combined |
| `PREMIUM-100` | 100 | 55 | 22 | 80 | 30 | Clean, AllIn |

Every family provides 35,040 UTC steps with Lisbon local-calendar attributes,
official 2023 Portuguese OMIE day-ahead prices, member demand, PV, weather,
carbon, EV sessions, deferrable requests, electrical-service limits and
community settlement. Member, physical-charger, EV and session identities are
kept separate, supporting multiple chargers per member and multiple EV sessions
per shared charger.

The data are reproducible calibrated scenarios for controlled algorithm
comparison. They are not statistically fitted samples of Portuguese energy
communities. Their electrical model represents active-power connection and
phase headroom; it is not a feeder voltage, reactive-power, protection or
power-flow model.

## Runtime and topology

- Derived load and PV forecasts can use causal previous-day persistence.
- OMIE price forecasts can respect the declared day-ahead publication boundary.
- `terminal_observation_padding` exposes one observation-only terminal boundary
  after all requested control transitions.
- EV service accounting distinguishes physical chargers, EVs and sessions,
  including back-to-back and terminal departures.
- Current-SOC telemetry initializes a connection boundary without overwriting
  the subsequent controlled SOC trajectory.
- Dynamic topology correctly initializes members and assets activated inside an
  episode or partial window, skips already expired deferrable requests and
  retains removed runtime instances for final KPI aggregation.
- Reinstalled stateful assets start from their declared initial condition rather
  than inheriting the removed instance's state.

## KPI and robustness semantics

Electrical-service reporting now distinguishes two concepts:

- `*_electrical_service_phase_requested_pressure_*` records controller-request
  pressure before projection; and
- `*_electrical_service_phase_violations_*` records residual applied-power
  exceedance after projection.

The legacy `charging_constraint_violation_kwh` observation and reward term keep
their pre-projection meaning. EV KPIs additionally report connected SOC gain and
energy-accounting shortfall. Dynamic KPIs aggregate historical charger, BESS and
deferrable instances, and asset-unavailability counts are deduplicated by
physical asset and time step.

## Reproducibility and validation

The release includes the annual-suite generator, frozen per-file checksums and
structural, scientific, diversity, deterministic-regeneration and execution
audits.

- Full test suite: `457 passed, 18 warnings`.
- Critical lint and Python 3.9 syntax targeting: pass.
- CI performance smoke: pass.
- Structural checks: 114 Micro, 204 Core-15, 358 Core-30 and 1,136 Premium
  family checks, plus five cross-family checks.
- Deterministic regeneration: all four checksum manifests byte-identical.
- Nine-schema smoke: 6,048 regular transitions, 120 topology-event effects and
  31,565 causal price features verified.

## Migration

Existing scenarios require no migration unless they opt into the new contracts.
Consumers that used electrical-service violation KPIs as a proxy for requested
action clipping should use the new requested-pressure KPIs. Algorithm
environments that rely on the annual suite or the new runtime/KPI behaviour
should pin `softcpsrecsimulator==1.8.0` and provide the repository dataset path
when the datasets are not mounted locally.
