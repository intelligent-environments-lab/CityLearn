#!/usr/bin/env python3
"""Run reproducible execution smokes for every canonical annual REC schema."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from citylearn.citylearn import CityLearnEnv


FAMILY_DIRECTORIES = (
    "rec_2023_micro_4_q",
    "rec_2023_core_15_stripped",
    "rec_2023_core_30",
    "rec_2023_premium_100",
)
DYNAMIC_VARIANTS = {
    "CORE-30-DYNAMIC",
    "CORE-30-COMBINED",
    "PREMIUM-100-CLEAN",
    "PREMIUM-100-ALLIN",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _zero_actions(env: CityLearnEnv) -> Mapping[str, Any]:
    return {
        "tables": {
            name: np.zeros(space.shape, dtype="float32")
            for name, space in env.action_space["tables"].items()
        }
    }


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all():
            raise AssertionError(f"{label} contains a non-finite value")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise AssertionError(f"{label} contains a non-finite scalar")


def _assert_publication_aware_price_features(
    env: CityLearnEnv,
    observations: Mapping[str, Any],
) -> int:
    """Independently verify the entity price bundle against causal inputs."""

    contract = (env.schema.get("derived_forecasts", {}) or {})
    if contract.get("price_source") != "publication_aware_day_ahead_market_input":
        return 0

    features = env.entity_specs["tables"]["district"]["features"]
    row = observations["tables"]["district"][0]
    pricing = env.buildings[0].pricing
    actual = np.asarray(pricing.electricity_pricing, dtype="float64")
    predicted = (
        np.asarray(pricing.electricity_pricing_predicted_1, dtype="float64"),
        np.asarray(pricing.electricity_pricing_predicted_2, dtype="float64"),
        np.asarray(pricing.electricity_pricing_predicted_3, dtype="float64"),
    )
    declared_horizons = tuple(int(value) for value in contract["price_horizon_steps"])
    period_steps = int(round(
        float(contract["persistence_period_seconds"])
        / float(env.seconds_per_time_step)
    ))
    time_step = int(env.time_step)
    requested = {
        label: max(int(np.ceil(seconds / float(env.seconds_per_time_step))), 1)
        for label, seconds in (
            ("15m", 900),
            ("1h", 3600),
            ("3h", 10_800),
            ("6h", 21_600),
            ("24h", 86_400),
        )
    }

    for label, steps_ahead in requested.items():
        expected = None
        for declared_horizon, values in zip(declared_horizons, predicted):
            if declared_horizon < steps_ahead:
                continue
            issue_step = time_step - (declared_horizon - steps_ahead)
            if issue_step >= 0:
                expected = float(values[issue_step])
                break
        if expected is None:
            target = min(time_step + steps_ahead, len(actual) - 1)
            source = target - period_steps
            source = time_step if source < 0 or source > time_step else source
            expected = float(actual[source])

        observed = float(row[features.index(f"forecast_price_next_{label}")])
        if not np.isclose(observed, expected, atol=1.0e-7):
            raise AssertionError(
                f"forecast_price_next_{label}={observed} but causal input is {expected} "
                f"at time step {time_step}"
            )
    return len(requested)


def _assert_new_topology_event_effects(
    env: CityLearnEnv,
    processed_events: int,
) -> tuple[int, int]:
    """Verify that newly applied topology events changed physical runtime state."""

    event_log = list(env.topology_event_log)
    checks = 0
    active_members = {building.name for building in env.buildings}
    topology_service = getattr(env, "_topology_service", None)
    member_pool = {} if topology_service is None else topology_service.member_pool

    def _assert_storage_activation_soc(building, label: str) -> None:
        storage = getattr(building, "electrical_storage", None)
        if (
            storage is None
            or float(getattr(storage, "capacity", 0.0) or 0.0) <= 0.0
            or float(getattr(storage, "nominal_power", 0.0) or 0.0) <= 0.0
        ):
            return
        expected = float(getattr(storage, "initial_soc", 0.0) or 0.0)
        expected = max(expected, float(getattr(storage, "_minimum_soc", lambda: 0.0)()))
        observed = float(storage.soc[storage.time_step])
        if not np.isclose(observed, expected, atol=1.0e-7):
            raise AssertionError(
                f"{label} storage SoC={observed}, declared activation SoC={expected}"
            )

    for event in event_log[processed_events:]:
        if not event.get("applied", False):
            raise AssertionError(f"topology event {event.get('id')} was not applied")

        operation = event["operation"]
        member_id = event.get("target_member_id")
        if operation == "add_member":
            if member_id not in active_members:
                raise AssertionError(f"added member {member_id} is not active")
            _assert_storage_activation_soc(member_pool[member_id], f"added member {member_id}")
            checks += 1
            continue
        if operation == "remove_member":
            if member_id in active_members:
                raise AssertionError(f"removed member {member_id} remains active")
            checks += 1
            continue

        building = member_pool.get(member_id)
        if building is None:
            raise AssertionError(f"event target member {member_id} is absent from the pool")

        asset_type = event.get("target_asset_type")
        asset_id = event.get("target_asset_id")
        added = operation == "add_asset"
        if asset_type == "charger":
            present = any(
                charger.charger_id == asset_id
                for charger in building.electric_vehicle_chargers or []
            )
            if present != added:
                raise AssertionError(
                    f"charger {member_id}/{asset_id} presence={present} after {operation}"
                )
        elif asset_type == "deferrable_appliance":
            present = any(
                appliance.name == asset_id
                for appliance in building.deferrable_appliances or []
            )
            if present != added:
                raise AssertionError(
                    f"deferrable {member_id}/{asset_id} presence={present} after {operation}"
                )
        elif asset_type == "electrical_storage":
            storage = building.electrical_storage
            present = (
                float(getattr(storage, "capacity", 0.0)) > 0.0
                and float(getattr(storage, "nominal_power", 0.0)) > 0.0
            )
            if present != added:
                raise AssertionError(
                    f"storage {member_id} presence={present} after {operation}"
                )
            if added:
                _assert_storage_activation_soc(building, f"restored storage {member_id}")
        elif asset_type == "pv":
            present = float(getattr(building.pv, "nominal_power", 0.0)) > 0.0
            if present != added:
                raise AssertionError(f"PV {member_id} presence={present} after {operation}")

            time_step = int(env.time_step)
            observed = float(building.solar_generation[time_step])
            if added:
                expected = float(
                    building._pv_generation_to_control_step(
                        [building.energy_simulation.solar_generation[time_step]]
                    )[0]
                    * -1.0
                )
                if not np.isclose(observed, expected, atol=1.0e-7):
                    raise AssertionError(
                        f"restored PV {member_id} produces {observed}, expected {expected}"
                    )
            elif not np.isclose(observed, 0.0, atol=1.0e-7):
                raise AssertionError(
                    f"removed PV {member_id} still produces {observed} kWh"
                )
        else:
            raise AssertionError(f"unsupported audited asset type {asset_type}")
        checks += 1

    return len(event_log), checks


def _rollout(schema: str | Mapping[str, Any], steps: int) -> Mapping[str, Any]:
    started = time.perf_counter()
    env = CityLearnEnv(
        schema,
        offline=True,
        # CityLearn counts the reset state as one episode time point. Request
        # one extra point so ``steps`` means completed control transitions.
        episode_time_steps=steps + 1,
        render_mode="none",
        random_seed=2023,
    )
    completed = 0
    reward_min = np.inf
    reward_max = -np.inf
    causal_price_feature_checks = 0
    topology_effect_checks = 0
    processed_topology_events = 0
    try:
        observations, _ = env.reset()
        _assert_finite(observations, "reset_observations")
        causal_price_feature_checks += _assert_publication_aware_price_features(
            env,
            observations,
        )
        processed_topology_events, new_checks = _assert_new_topology_event_effects(
            env,
            processed_topology_events,
        )
        topology_effect_checks += new_checks
        while completed < steps:
            observations, rewards, terminated, truncated, _ = env.step(
                _zero_actions(env)
            )
            _assert_finite(observations, f"observations_step_{completed + 1}")
            causal_price_feature_checks += _assert_publication_aware_price_features(
                env,
                observations,
            )
            processed_topology_events, new_checks = _assert_new_topology_event_effects(
                env,
                processed_topology_events,
            )
            topology_effect_checks += new_checks
            _assert_finite(rewards, f"rewards_step_{completed + 1}")
            reward_values = np.asarray(rewards, dtype="float64")
            reward_min = min(reward_min, float(reward_values.min()))
            reward_max = max(reward_max, float(reward_values.max()))
            completed += 1
            if terminated or truncated:
                break
        return {
            "status": "pass",
            "steps_completed": completed,
            "reward_minimum": round(reward_min, 6),
            "reward_maximum": round(reward_max, 6),
            "final_time_step": int(env.time_step),
            "topology_version": int(env.topology_version),
            "topology_events_applied": len(env.topology_event_log),
            "topology_effect_checks": topology_effect_checks,
            "causal_price_feature_checks": causal_price_feature_checks,
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        env.close()


def _compressed_dynamic_schema(
    schema_path: Path,
    root: Path,
) -> tuple[Mapping[str, Any], int]:
    schema = dict(_read_json(schema_path))
    schema["root_directory"] = str(root.resolve())
    events = [dict(event) for event in schema.get("topology_events", [])]
    for index, event in enumerate(events, start=1):
        event["time_step"] = index * 2
    schema["topology_events"] = events
    steps = max((event["time_step"] for event in events), default=0) + 3
    return schema, steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = REPOSITORY
    parser.add_argument(
        "--dataset-root", type=Path, default=repository / "data/datasets"
    )
    parser.add_argument("--rollout-steps", type=int, default=672)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=repository / "results/raw/annual_rec_suite_smoke_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regular = {}
    dynamic = {}
    for directory in FAMILY_DIRECTORIES:
        family_root = args.dataset_root / directory
        manifest = _read_json(family_root / "dataset_manifest.json")
        for variant, relative in manifest["variants"].items():
            schema_path = family_root / relative
            print(f"weekly rollout: {variant}")
            regular[variant] = _rollout(str(schema_path), args.rollout_steps)
            if variant in DYNAMIC_VARIANTS:
                schema, steps = _compressed_dynamic_schema(schema_path, family_root)
                print(f"compressed topology timeline: {variant}")
                dynamic[variant] = _rollout(schema, steps)
                expected_events = len(schema.get("topology_events", []))
                if dynamic[variant]["topology_events_applied"] != expected_events:
                    raise AssertionError(
                        f"{variant} applied {dynamic[variant]['topology_events_applied']} "
                        f"of {expected_events} compressed topology events"
                    )
    report = {
        "audit_date": date.today().isoformat(),
        "status": "pass",
        "rollout_steps_per_schema": args.rollout_steps,
        "regular_schema_rollouts": regular,
        "compressed_dynamic_timeline_rollouts": dynamic,
        "summary": {
            "schemas": len(regular),
            "regular_steps_completed": sum(item["steps_completed"] for item in regular.values()),
            "dynamic_schemas": len(dynamic),
            "compressed_topology_events_applied": sum(
                item["topology_events_applied"] for item in dynamic.values()
            ),
            "topology_effect_checks": sum(
                item["topology_effect_checks"]
                for item in [*regular.values(), *dynamic.values()]
            ),
            "causal_price_feature_checks": sum(
                item["causal_price_feature_checks"]
                for item in [*regular.values(), *dynamic.values()]
            ),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
