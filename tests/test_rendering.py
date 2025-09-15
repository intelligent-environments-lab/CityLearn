"""
Run from tests folder: python3 test_rendering.py

Ensures parent repo root is on sys.path so local 'citylearn' package is importable
without installing. Alternative: run from repo root using `python -m tests.test_rendering`.

Validates rendering switch and export behavior:
- Rendering disabled: no output folder created during steps.
- Rendering enabled: output folder created.
- export_final_kpis writes file even if rendering disabled.

Set CLEAN_RESULTS=1 env var to remove created result folders at end.
"""
import os
import sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import shutil
import numpy as np

from citylearn.citylearn import CityLearnEnv


def maybe_cleanup(path: str):
    if path and os.environ.get('CLEAN_RESULTS', '0') == '1':
        try:
            shutil.rmtree(path)
            print(f'Removed {path}')
        except Exception as e:
            print(f'Cleanup failed for {path}: {e}')


def main():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'

    # 1) Rendering disabled (default)
    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=2)
    obs, _ = env.reset()
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    obs, r, term, trunc, _ = env.step(zeros)
    has_folder = hasattr(env, 'new_folder_path') and os.path.isdir(getattr(env, 'new_folder_path', ''))
    assert not has_folder, 'Results folder present while rendering disabled.'

    # 2) Rendering enabled via param
    env2 = CityLearnEnv(schema, central_agent=True, episode_time_steps=2, render=True)
    obs, _ = env2.reset()
    obs, r, term, trunc, _ = env2.step(zeros)
    assert hasattr(env2, 'new_folder_path'), 'new_folder_path missing with rendering enabled.'
    assert os.path.isdir(env2.new_folder_path), f'Results folder not created: {env2.new_folder_path}'

    # 3) export_final_kpis works even when rendering disabled (Option B)
    class _Model: pass
    m = _Model(); m.env = env
    export_name = 'exported_kpis_test.csv'
    env.export_final_kpis(m, filepath=export_name)
    assert hasattr(env, 'new_folder_path'), 'export did not set new_folder_path.'
    export_path = os.path.join(env.new_folder_path, export_name)
    assert os.path.isfile(export_path), f'export file missing: {export_path}'

    print('OK - rendering off/on and export checks passed.')

    maybe_cleanup(getattr(env2, 'new_folder_path', None))
    maybe_cleanup(getattr(env, 'new_folder_path', None))


if __name__ == '__main__':
    main()
