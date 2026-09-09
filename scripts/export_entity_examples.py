#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from citylearn.citylearn import CityLearnEnv


def _zero_entity_actions(env: CityLearnEnv) -> Mapping[str, Mapping[str, np.ndarray]]:
    tables = env.action_space["tables"]
    return {
        "tables": {
            "building": np.zeros(tables["building"].shape, dtype=np.float32),
            "charger": np.zeros(tables["charger"].shape, dtype=np.float32),
        }
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return value


def _serialize_space(space: spaces.Space) -> Mapping[str, Any]:
    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                str(k): _serialize_space(v) for k, v in space.spaces.items()
            },
        }

    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": _to_jsonable(space.low),
            "high": _to_jsonable(space.high),
        }

    return {
        "type": space.__class__.__name__,
        "repr": repr(space),
    }


def _write_json(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def export_examples(schema: Path, output_dir: Path, target_time_step: int, seed: int):
    env = CityLearnEnv(str(schema), interface="entity", random_seed=seed)
    try:
        observations, _ = env.reset(seed=seed)

        while (
            env.time_step < target_time_step
            and not env.terminated
            and not env.truncated
        ):
            observations, *_ = env.step(_zero_entity_actions(env))

        suffix = f"t{int(env.time_step)}"
        run_meta = {
            "schema": str(schema.resolve()),
            "time_step": int(env.time_step),
            "topology_version": int(getattr(env, "topology_version", 0)),
            "seed": int(seed),
            "terminated": bool(env.terminated),
            "truncated": bool(env.truncated),
        }

        _write_json(
            output_dir / f"entity_observations_{suffix}.json",
            {
                "meta": run_meta,
                "payload": observations,
            },
        )
        _write_json(
            output_dir / f"entity_specs_{suffix}.json",
            {
                "meta": run_meta,
                "payload": env.entity_specs,
            },
        )
        _write_json(
            output_dir / f"entity_action_space_{suffix}.json",
            {
                "meta": run_meta,
                "payload": _serialize_space(env.action_space),
            },
        )
        _write_json(
            output_dir / f"entity_observation_space_{suffix}.json",
            {
                "meta": run_meta,
                "payload": _serialize_space(env.observation_space),
            },
        )

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export real entity payload/spec/space examples into JSON files."
    )
    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="Path to dataset schema.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/entity_examples"),
        help="Directory where JSON examples will be written.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=2200,
        help="Target environment time step for the exported snapshot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    export_examples(
        schema=args.schema,
        output_dir=args.output_dir,
        target_time_step=args.time_step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
