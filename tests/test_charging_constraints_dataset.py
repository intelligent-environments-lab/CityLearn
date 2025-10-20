"""Integration test for the charging constraints demo dataset."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


DATASET_PATH = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_charging_constraints_demo/schema.json"


def _find_action_index(env, charger_id: str) -> int:
    names = env.action_names[0]
    return names.index(f'electric_vehicle_storage_{charger_id}')


def test_charging_constraints_demo_dataset_runs():
    env = CityLearnEnv(str(DATASET_PATH), central_agent=True, episode_time_steps=4, random_seed=1)

    try:
        env.reset()
        charger_ids = ['charger_15_1', 'charger_15_2']
        actions = np.zeros(len(env.action_names[0]), dtype='float32')
        for cid in charger_ids:
            actions[_find_action_index(env, cid)] = 1.0

        reward = env.step([actions])[1]
        assert reward is not None

        building = next(b for b in env.buildings if b.name == 'Building_15')
        obs = building.observations(include_all=True, normalize=False, periodic_normalization=False)

        assert 'charging_constraint_violation_kwh' in obs
        assert obs['charging_constraint_violation_kwh'] >= 0.0
        one_hot_keys = [k for k in obs if k.startswith('charging_phase_one_hot_charger_15_1_')]
        assert one_hot_keys
        assert abs(sum(obs[k] for k in one_hot_keys) - 1.0) < 1e-6

    finally:
        env.close()


if __name__ == '__main__':
    test_charging_constraints_demo_dataset_runs()
    print('Charging constraints demo dataset test completed successfully.')
