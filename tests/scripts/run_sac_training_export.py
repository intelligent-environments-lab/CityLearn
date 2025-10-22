#!/usr/bin/env python3
"""Run a short SAC training episode and verify automatic exports."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.citylearn import CityLearnEnv  # noqa: E402

SCHEMA = "baeda_3dem"


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)

    render_root = ROOT / "SimulationData"
    env = CityLearnEnv(
        SCHEMA,
        episode_time_steps=96,
        render_mode="end",
        render_directory=render_root,
        render_session_name="sac_export_example",
        random_seed=0,
    )

    try:
        agent = env.load_agent()
        agent.learn(episodes=1, deterministic_finish=False, logging_level=logging.INFO)
        outputs_path = Path(env.new_folder_path)
        print(f"Exports written to: {outputs_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
