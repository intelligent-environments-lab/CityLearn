"""Rendering/export behaviour tests (pytest version)."""

import csv
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
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


def test_none_mode_does_not_auto_export_kpis_by_default(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=4,
        render_mode="none",
        render_directory=tmp_path,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        while not env.terminated:
            _, _, terminated, truncated, _ = env.step(zeros)
            if terminated or truncated:
                break

        assert env.new_folder_path is None
        assert not env._final_kpis_exported
    finally:
        _cleanup_env(env)
        env.close()


def test_none_mode_can_auto_export_kpis_when_enabled(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=4,
        render_mode="none",
        render_directory=tmp_path,
        export_kpis_on_episode_end=True,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        while not env.terminated:
            _, _, terminated, truncated, _ = env.step(zeros)
            if terminated or truncated:
                break

        outputs_path = Path(env.new_folder_path)
        assert outputs_path.is_dir()
        assert (outputs_path / "exported_kpis.csv").is_file()
        assert env._final_kpis_exported
    finally:
        _cleanup_env(env)
        env.close()


def test_during_mode_can_disable_auto_kpi_export(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=4,
        render_mode="during",
        render_directory=tmp_path,
        export_kpis_on_episode_end=False,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        while not env.terminated:
            _, _, terminated, truncated, _ = env.step(zeros)
            if terminated or truncated:
                break

        outputs_path = Path(env.new_folder_path)
        assert (outputs_path / "exported_data_community_ep0.csv").is_file()
        assert not (outputs_path / "exported_kpis.csv").exists()
        assert not env._final_kpis_exported
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

        assert int(year_str) == env.render_start_date.year == 2024
        assert int(day_str) == env.render_start_date.day
        assert int(month_str) == env.render_start_date.month
        assert time_part == "00:00:00"
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

        assert (int(year_str), int(month_str), int(day_str)) == (2026, 5, 15)
    finally:
        env.close()


def test_render_timestamp_advances_with_seconds_per_time_step():
    schema = _load_schema_dict()
    schema.pop('start_date', None)

    env = CityLearnEnv(
        schema,
        central_agent=True,
        episode_time_steps=4,
        seconds_per_time_step=60,
    )

    try:
        env.reset()
        t0 = env._get_iso_timestamp()
        env.step([np.zeros(env.action_space[0].shape[0], dtype="float32")])
        t1 = env._get_iso_timestamp()
        assert t0.endswith("00:00:00")
        assert t1.endswith("00:01:00")
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

        # Header + one row per simulated timestep
        assert len(rows) == env.time_step + 1

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

        assert len(rows) == env.time_step + 1
    finally:
        _cleanup_env(env)
        env.close()


def test_end_mode_exports_without_per_step_buffer_growth(tmp_path):
    env = CityLearnEnv(
        str(DATASET),
        central_agent=True,
        episode_time_steps=5,
        render_mode="end",
        render_directory=tmp_path,
        random_seed=0,
    )

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]

        while not env.terminated:
            _, _, terminated, truncated, _ = env.step(zeros)
            assert not any(env._render_buffer.values())
            if terminated or truncated:
                break

        outputs_path = Path(env.new_folder_path)
        community_file = outputs_path / "exported_data_community_ep0.csv"
        assert community_file.is_file()

        with community_file.open(newline="") as handle:
            rows = list(csv.reader(handle))

        # Header + one row per realized transition.
        assert len(rows) == env.time_step + 1
    finally:
        _cleanup_env(env)
        env.close()


def test_end_mode_export_file_contract_matches_during_mode(tmp_path):
    def _run(render_mode: str):
        env = CityLearnEnv(
            str(DATASET),
            central_agent=True,
            episode_time_steps=4,
            render_mode=render_mode,
            render_directory=tmp_path / render_mode,
            random_seed=0,
        )
        try:
            env.reset()
            zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
            while not env.terminated:
                _, _, terminated, truncated, _ = env.step(zeros)
                if terminated or truncated:
                    break

            outputs_path = Path(env.new_folder_path)
            export_files = sorted(p.name for p in outputs_path.glob("exported_data_*_ep0.csv"))
            community_file = outputs_path / "exported_data_community_ep0.csv"
            with community_file.open(newline="") as handle:
                header = next(csv.reader(handle))

            return export_files, header
        finally:
            _cleanup_env(env)
            env.close()

    during_files, during_header = _run("during")
    end_files, end_header = _run("end")

    assert end_files == during_files
    assert end_header == during_header


def test_end_mode_ev_and_charger_content_matches_during_mode(tmp_path):
    def _run(render_mode: str) -> Path:
        env = CityLearnEnv(
            str(DATASET),
            central_agent=True,
            episode_time_steps=128,
            render_mode=render_mode,
            render_directory=tmp_path / render_mode,
            random_seed=7,
        )
        agent = Agent(env)

        try:
            observations, _ = env.reset()

            while not env.terminated:
                actions = agent.predict(observations, deterministic=True)
                observations, _, terminated, truncated, _ = env.step(actions)

                if terminated or truncated:
                    break

            return Path(env.new_folder_path)
        finally:
            env.close()

    during_dir = _run("during")
    end_dir = _run("end")

    all_files = sorted(p.name for p in during_dir.glob("exported_data_*_ep0.csv"))
    relevant_files = [name for name in all_files if ("charger" in name or "electric_vehicle" in name)]
    assert relevant_files, "Expected EV/charger export files to be present."

    for filename in relevant_files:
        during_path = during_dir / filename
        end_path = end_dir / filename
        assert end_path.exists(), f"Missing end-mode file: {filename}"

        with during_path.open(newline="") as handle:
            during_rows = list(csv.DictReader(handle))

        with end_path.open(newline="") as handle:
            end_rows = list(csv.DictReader(handle))

        assert during_rows == end_rows, f"Mismatch in EV/charger export data for file {filename}"
