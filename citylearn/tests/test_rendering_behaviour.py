"""Rendering/export behaviour tests (pytest version)."""

import json
from pathlib import Path
import shutil
import numpy as np

from citylearn.citylearn import CityLearnEnv


DATASET = Path(__file__).resolve().parents[2] / 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'


def _load_schema_dict() -> dict:
    with DATASET.open() as f:
        schema = json.load(f)

    schema['root_directory'] = str(DATASET.parent)
    return schema


def _cleanup_env(env: CityLearnEnv):
    path = getattr(env, 'new_folder_path', None)
    if path:
        shutil.rmtree(path, ignore_errors=True)


def _step_once(env: CityLearnEnv):
    env.reset()
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    env.step(zeros)


def test_render_disabled_leaves_no_directory():
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=2)
    try:
        _step_once(env)
        assert getattr(env, 'new_folder_path', None) is None
    finally:
        _cleanup_env(env)
        env.close()


def test_render_enabled_creates_default_directory():
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=2, render=True)
    try:
        _step_once(env)
        new_folder = getattr(env, 'new_folder_path', None)
        assert new_folder, 'new_folder_path missing with rendering enabled.'
        new_folder_path = Path(new_folder)
        assert new_folder_path.is_dir()
        assert env.render_output_root.resolve() in new_folder_path.resolve().parents
    finally:
        _cleanup_env(env)
        env.close()


def test_export_final_kpis_when_render_off(tmp_path):
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=2, render_directory=tmp_path)

    class _Model:
        pass

    model = _Model()
    model.env = env

    try:
        env.export_final_kpis(model, filepath='exported_kpis_test.csv')
        new_folder = getattr(env, 'new_folder_path', None)
        assert new_folder
        export_path = Path(new_folder) / 'exported_kpis_test.csv'
        assert export_path.is_file()
    finally:
        _cleanup_env(env)
        env.close()


def test_render_directory_override(tmp_path):
    custom_root = tmp_path / 'custom_results'

    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=2,
        render=True,
        render_directory=custom_root,
    )

    try:
        _step_once(env)
        new_folder = Path(env.new_folder_path)
        assert env.render_output_root == custom_root.resolve()
        assert new_folder.is_dir()
        assert env.render_output_root in new_folder.parents
    finally:
        _cleanup_env(env)
        env.close()


def test_default_start_date_used_for_render_timestamp():
    schema = _load_schema_dict()
    schema.pop('start_date', None)

    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=2)

    try:
        env.reset()
        timestamp = env._get_iso_timestamp()
        date_part, time_part = timestamp.split('T')
        year_str, month_str, day_str = date_part.split('-')

        energy_sim = env.buildings[0].energy_simulation
        assert int(year_str) == env.render_start_date.year == 2024
        assert int(day_str) == env.render_start_date.day == 1
        assert int(month_str) == int(energy_sim.month[0])

        expected_hour = int(energy_sim.hour[0]) % 24
        expected_minutes = 0
        minutes_data = getattr(energy_sim, 'minutes', None)
        if minutes_data is not None and len(minutes_data) > 0:
            expected_minutes = int(minutes_data[0])

        assert time_part == f"{expected_hour:02d}:{expected_minutes:02d}:00"
    finally:
        env.close()


def test_schema_start_date_overrides_default_timestamp_start():
    schema = _load_schema_dict()
    schema['start_date'] = '2026-05-15'

    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=2)

    try:
        env.reset()
        timestamp = env._get_iso_timestamp()
        date_part, _ = timestamp.split('T')
        year_str, month_str, day_str = date_part.split('-')

        energy_sim = env.buildings[0].energy_simulation
        assert (int(year_str), int(day_str)) == (2026, 15)
        assert int(month_str) == int(energy_sim.month[0])
    finally:
        env.close()
