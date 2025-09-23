"""Rendering/export behaviour tests (pytest version)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import numpy as np

from citylearn.citylearn import CityLearnEnv


DATASET = Path(__file__).resolve().parents[2] / 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'


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
