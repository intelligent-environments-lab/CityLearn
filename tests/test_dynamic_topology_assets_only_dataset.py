from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json"
)


def _load_schema() -> Mapping[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text())
    schema["root_directory"] = str(SCHEMA_PATH.parent)
    schema["interface"] = "entity"
    schema["topology_mode"] = "dynamic"

    # Keep event ordering but bring events early so tests stay fast.
    remapped_steps = [2, 3, 4, 5, 6, 7]
    for event, step in zip(schema.get("topology_events", []), remapped_steps):
        event["time_step"] = step

    return schema


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            name: np.zeros(space.shape, dtype="float32")
            for name, space in tables.items()
            if name in {"building", "charger", "deferrable_appliance"}
        }
    }


def _asset_counts(env: CityLearnEnv) -> Mapping[str, int]:
    return {
        "buildings": len(env.entity_specs["tables"]["building"]["ids"]),
        "chargers": len(env.entity_specs["tables"]["charger"]["ids"]),
        "pv": len(env.entity_specs["tables"]["pv"]["ids"]),
        "storage": len(env.entity_specs["tables"]["storage"]["ids"]),
    }


def test_assets_only_dynamic_topology_churn_and_contract():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=12, random_seed=0)

    try:
        env.reset(seed=0)

        assert len(env.buildings) == 17
        assert all(building.name != "Building_18" for building in env.buildings)

        specs_blob = json.dumps(env.entity_specs)
        assert "washing_machine" not in specs_blob

        timeline = {}
        for _ in range(8):
            timeline[env.time_step] = dict(_asset_counts(env))
            env.step(_zero_entity_actions(env))

        assert all(values["buildings"] == 17 for values in timeline.values())

        assert timeline[0] == {"buildings": 17, "chargers": 8, "pv": 16, "storage": 16}
        assert timeline[2]["chargers"] - timeline[1]["chargers"] == 1
        assert timeline[3]["chargers"] - timeline[2]["chargers"] == -1
        assert timeline[4]["chargers"] - timeline[3]["chargers"] == 1
        assert timeline[7]["chargers"] - timeline[6]["chargers"] == 1
        assert timeline[5]["pv"] - timeline[4]["pv"] == 1
        assert timeline[6]["storage"] - timeline[5]["storage"] == 1
        assert timeline[7] == {"buildings": 17, "chargers": 10, "pv": 17, "storage": 17}

        event_log = env.topology_event_log
        assert len(event_log) == 6

        operations = [event["operation"] for event in event_log]
        assert set(operations) == {"add_asset", "remove_asset"}
        assert operations.count("add_asset") == 5
        assert operations.count("remove_asset") == 1
    finally:
        env.close()
