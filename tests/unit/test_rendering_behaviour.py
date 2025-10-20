"""Rendering/export behaviour tests (pytest version)."""

import csv
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


DATASET = Path(__file__).resolve().parents[2] / 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'


def _load_schema_dict() -> dict:
    with DATASET.open() as f:
        schema = json.load(f)

    schema['root_directory'] = str(DATASET.parent)
    return schema


def _cleanup_env(env: CityLearnEnv):
    paths = set()
    outputs = getattr(env, 'new_folder_path', None)
    if outputs:
        paths.add(Path(outputs))

    session_dir = getattr(env, '_render_session_dir', None)
    if session_dir:
        paths.add(Path(session_dir))

    for path in paths:
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
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=2, render_mode="during")
    try:
        _step_once(env)

        outputs_path = Path(env.new_folder_path)
        assert outputs_path.is_dir()
        assert outputs_path.parent == env.render_output_root
        assert (outputs_path / 'exported_data_community_ep0.csv').is_file()
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
        outputs_path = Path(env.new_folder_path)
        assert (outputs_path / 'exported_kpis_test.csv').is_file()
    finally:
        _cleanup_env(env)
        env.close()


def test_render_directory_override(tmp_path):
    custom_root = tmp_path / 'custom_results'

    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=2,
        render_mode="during",
        render_directory=custom_root,
    )

    try:
        _step_once(env)
        outputs_path = Path(env.new_folder_path)
        assert env.render_output_root == custom_root.resolve()
        assert outputs_path.is_dir()
        assert outputs_path.parent == env.render_output_root
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
        first_hour = int(energy_sim.hour[0])
        month_series = energy_sim.month

        if first_hour >= 24:
            expected_month = int(month_series[1]) if len(month_series) > 1 else ((int(month_series[0]) % 12) + 1)
            expected_day = 1
        else:
            expected_month = int(month_series[0])
            expected_day = env.render_start_date.day

        assert int(year_str) == env.render_start_date.year == 2024
        assert int(day_str) == expected_day
        assert int(month_str) == expected_month

        expected_hour = first_hour % 24
        minutes_data = getattr(energy_sim, 'minutes', None)
        expected_minutes = int(minutes_data[0]) if minutes_data is not None and len(minutes_data) > 0 else 0

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
        first_hour = int(energy_sim.hour[0])
        month_series = energy_sim.month

        if first_hour >= 24:
            expected_month = int(month_series[1]) if len(month_series) > 1 else ((int(month_series[0]) % 12) + 1)
            expected_day = 1
        else:
            expected_month = int(month_series[0])
            expected_day = 15

        assert (int(year_str), int(day_str)) == (2026, expected_day)
        assert int(month_str) == expected_month
    finally:
        env.close()


def test_render_after_episode_completion(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=4,
        render_mode="during",
        render_directory=tmp_path,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        while not env.terminated:
            env.step(zeros)

        outputs_path = Path(env.new_folder_path)
        assert (outputs_path / 'exported_data_community_ep0.csv').exists()
    finally:
        _cleanup_env(env)
        env.close()


def test_render_mid_and_end_exports(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=4,
        render_mode="end",
        render_directory=tmp_path,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        mid_step = env.episode_tracker.episode_time_steps // 2

        for step in range(env.episode_tracker.episode_time_steps):
            _, _, terminated, truncated, _ = env.step(zeros)

            if step == mid_step:
                env.render()

            if terminated or truncated:
                break

        outputs_path = Path(env.new_folder_path)
        community_file = outputs_path / 'exported_data_community_ep0.csv'
        assert community_file.is_file()
        assert any(outputs_path.glob('exported_data_*_ep0.csv'))

        with community_file.open(newline='') as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        # Header + one row per timestep
        assert len(rows) == env.episode_tracker.episode_time_steps + 1

        class _Model:
            pass

        model = _Model()
        model.env = env
        env.export_final_kpis(model)

        assert (outputs_path / 'exported_kpis.csv').is_file()
    finally:
        _cleanup_env(env)
        env.close()


def test_export_final_kpis_flushes_end_mode(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=3,
        render_mode="end",
        render_directory=tmp_path,
        random_seed=0,
    )

    class _Model:
        pass

    model = _Model()
    model.env = env

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        while not env.terminated:
            env.step(zeros)

        env.export_final_kpis(model, filepath="exported_kpis_end.csv")

        outputs_path = Path(env.new_folder_path)
        community_file = outputs_path / "exported_data_community_ep0.csv"
        assert community_file.is_file()
        assert (outputs_path / "exported_kpis_end.csv").is_file()

        with community_file.open(newline="") as handle:
            rows = list(csv.reader(handle))

        assert len(rows) == env.episode_tracker.episode_time_steps + 1
    finally:
        _cleanup_env(env)
        env.close()
