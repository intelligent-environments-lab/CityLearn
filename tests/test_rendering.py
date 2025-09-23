"""
Run from repo root: `python tests/test_rendering.py`

Validates rendering/export behaviour without relying on pytest so it matches the
rest of the scripts in this folder.
"""

import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.citylearn import CityLearnEnv

SCHEMA = ROOT / 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'


def _cleanup_env(env: CityLearnEnv):
    path = getattr(env, 'new_folder_path', None)
    if path:
        shutil.rmtree(path, ignore_errors=True)


def _step_once(env: CityLearnEnv):
    obs, _ = env.reset()
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    env.step(zeros)


def test_render_disabled():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=2)
    try:
        _step_once(env)
        assert getattr(env, 'new_folder_path', None) is None, 'Results folder present while rendering disabled.'
    finally:
        _cleanup_env(env)


def test_render_enabled_default_directory():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=2, render=True)
    try:
        _step_once(env)
        new_folder = getattr(env, 'new_folder_path', None)
        assert new_folder, 'new_folder_path missing with rendering enabled.'
        new_folder_path = Path(new_folder)
        assert new_folder_path.is_dir(), f'Results folder not created: {new_folder_path}'
        assert env.render_output_root.resolve() in new_folder_path.resolve().parents, 'Render output not stored under render_output_root.'
    finally:
        _cleanup_env(env)


def test_export_final_kpis_when_render_off():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=2)

    class _Model:
        pass

    model = _Model()
    model.env = env

    try:
        env.export_final_kpis(model, filepath='exported_kpis_test.csv')
        new_folder = getattr(env, 'new_folder_path', None)
        assert new_folder, 'export_final_kpis did not create an output directory.'
        export_path = Path(new_folder) / 'exported_kpis_test.csv'
        assert export_path.is_file(), f'exported KPIs missing: {export_path}'
    finally:
        _cleanup_env(env)


def test_render_directory_override():
    with TemporaryDirectory() as tmpdir:
        custom_root = Path(tmpdir) / 'custom_results'
        env = CityLearnEnv(
            str(SCHEMA),
            central_agent=True,
            episode_time_steps=2,
            render=True,
            render_directory=custom_root,
        )
        try:
            _step_once(env)
            new_folder = Path(env.new_folder_path)
            assert env.render_output_root == custom_root.resolve(), 'render_directory override not respected.'
            assert new_folder.is_dir(), 'Custom render output directory missing.'
            assert env.render_output_root in new_folder.parents, 'Custom results stored outside provided directory.'
        finally:
            _cleanup_env(env)


def main():
    tests = [
        ('render disabled leaves no directory', test_render_disabled),
        ('render enabled creates timestamp directory', test_render_enabled_default_directory),
        ('export_final_kpis creates output when render off', test_export_final_kpis_when_render_off),
        ('render_directory override respected', test_render_directory_override),
    ]

    for description, func in tests:
        func()
        print(f'OK - {description}.')

    print('All rendering checks passed.')


if __name__ == '__main__':
    main()
