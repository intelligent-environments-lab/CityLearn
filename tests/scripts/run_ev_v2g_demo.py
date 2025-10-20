#!/usr/bin/env python3
"""Minimal EV V2G demonstration that produces an export with a discharge event."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.citylearn import CityLearnEnv  # noqa: E402

SCHEMA = ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def main() -> None:
    render_root = ROOT / "SimulationData"
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=8,
        render_mode="end",
        render_directory=render_root,
        render_session_name="ev_v2g_demo",
        random_seed=0,
    )

    try:
        _, _ = env.reset()
        action_size = env.action_space[0].shape[0]
        zero_action = np.zeros(action_size, dtype="float32")

        try:
            ev_index = next(
                idx
                for idx, name in enumerate(env.action_names[0])
                if name.startswith("electric_vehicle_storage_")
            )
        except StopIteration as exc:  # pragma: no cover
            raise RuntimeError("No EV storage action found in the environment action space.") from exc

        charge_action = zero_action.copy()
        charge_action[ev_index] = 1.0
        env.step([charge_action])

        discharge_action = zero_action.copy()
        discharge_action[ev_index] = -0.6
        env.step([discharge_action])

        while not env.terminated:
            env.step([zero_action])

        class _Adapter:
            def __init__(self, env: CityLearnEnv):
                self.env = env

        env.export_final_kpis(_Adapter(env))
        print(f"Exports written to: {env.new_folder_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
