"""
Run from tests folder: python3 test_rendering_end.py

Verifies that calling export routines at the end of a run creates the render
output directory in the expected location, even when rendering was disabled
during the simulation steps.
"""

import os
import sys
from pathlib import Path

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import numpy as np

from citylearn.citylearn import CityLearnEnv


def main():
    dataset_name = '../data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'
    custom_dir = Path('tests/tmp/render_end')

    env = CityLearnEnv(
        dataset_name,
        central_agent=True,
        render=False,
        episode_time_steps=2,
        render_directory=custom_dir,
    )

    obs, _ = env.reset()
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    obs, reward, terminated, truncated, info = env.step(zeros)

    print(f"Render enabled during steps? {env.render_enabled}")
    print(f"new_folder_path before export: {getattr(env, 'new_folder_path', None)}")

    class _Model:
        pass

    model = _Model()
    model.env = env

    export_name = 'final_kpis_end_test.csv'
    env.export_final_kpis(model, filepath=export_name)

    output_path = getattr(env, 'new_folder_path', None)
    if output_path:
        output_path = Path(output_path)
        print(f"Render directory created at: {output_path}")
        print(f"Exists? {output_path.is_dir()}")
        print(f"Custom base directory: {custom_dir.resolve()}")
        export_file = output_path / export_name
        print(f"Export file written? {export_file.is_file()} -> {export_file}")
    else:
        print('Render directory was not created.')


if __name__ == '__main__':
    main()
