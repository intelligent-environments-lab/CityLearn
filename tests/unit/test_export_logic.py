import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[2] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def test_save_to_csv_expands_headers(tmp_path):
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=2,
        render_directory=tmp_path,
        render_mode="during",
        random_seed=0,
    )

    try:
        env.render_output_root = tmp_path
        env._ensure_render_output_dir()

        env._save_to_csv("custom.csv", {"timestamp": "t0", "col_a": 1})
        env._save_to_csv("custom.csv", {"timestamp": "t1", "col_a": 2, "col_b": 3})

        csv_path = Path(env.new_folder_path) / "custom.csv"
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            rows = list(reader)

        assert headers == ["timestamp", "col_a", "col_b"]
        assert rows[0]["col_b"] == ""
        assert rows[1]["col_b"] == "3"
    finally:
        env.close()


def test_render_handles_missing_optional_components(tmp_path):
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=2,
        render_directory=tmp_path,
        render_mode="during",
        random_seed=0,
    )

    try:
        env.render_enabled = True
        env.render_output_root = tmp_path
        env.reset()

        env.buildings[0]._Building__electric_vehicle_chargers = []  # type: ignore[attr-defined]

        class DummyStorage:
            def __init__(self, length: int):
                self.energy_balance = np.zeros(length, dtype="float32")

            def as_dict(self):
                return {"placeholder": 0.0}

        env.buildings[0]._Building__electrical_storage = DummyStorage(env.episode_tracker.episode_time_steps)  # type: ignore[attr-defined]

        env.render()

        render_dir = Path(env.new_folder_path)
        community_exports = list(render_dir.glob("exported_data_community_ep*.csv"))
        assert community_exports, "Expected community export file to be created."
    finally:
        env.close()
