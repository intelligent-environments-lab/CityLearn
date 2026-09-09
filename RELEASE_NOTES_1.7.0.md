# softcpsrecsimulator 1.7.0 release notes

This fork publishes the release from the namespaced tag
`softcpsrecsimulator-v1.7.0`. The inherited upstream repository already owns
the unrelated historical tag `v1.7.0` from 2023; it is deliberately left
untouched.

## Entity runtime evidence

This release adds two backward-compatible subcontracts to the existing
`entity_v1` interface.

### `runtime_status_v1`

Entity observations now include `meta.runtime_status`. It exposes raw runtime
facts for typed and health-aware controllers:

- active fault/event causes and durations;
- actual charger-to-EV connection state;
- asset availability;
- sensor-channel status;
- actuator-channel status;
- community/cloud communication-link status; and
- value-quality evidence and freshness.

The Simulator does **not** derive a controller `HealthState`. `fault_mode`
remains the original cause: for example, `stuck` does not automatically mean
`STALE`. Policies and interface compilers must combine cause, duration,
freshness, semantic type and criticality according to their own versioned
rules.

Asset disconnection, asset unavailability, sensor loss, actuator loss and
communication loss remain separate facts. Normal EV disconnection is not an
equipment failure.

Existing robustness event files remain valid. They may optionally provide an
`event_domain` column. Omitting it uses a backward-compatible evidence-domain
default; it never emits health.

### `entity_action_execution_v1`

Entity-mode `step()` information now includes an action audit with stable
ownership and, where observable:

- requested action;
- action after channel faults;
- equipment-limited action;
- applied action/power; and
- limitation reasons.

Unavailable physical values are reported as `None`, not estimated.
`info.topology_events_applied` identifies topology events applied before the
returned observation.

## Compatibility

- `entity_v1` table, edge, feature and action meanings are unchanged.
- New observation metadata are additive and can be ignored by existing agents.
- Flat-interface inputs and outputs are unchanged.
- `entity_specs` now documents the runtime-status and action-execution
  subcontracts.

## Validation

The release includes regression and contract tests covering old event files,
fault-cause preservation, independent status domains, connection evidence,
requested-versus-applied actions, dynamic topology, entity observation bundles
and the legacy entity contract.
