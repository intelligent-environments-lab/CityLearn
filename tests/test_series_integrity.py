"""
Run from tests folder: python3 test_series_integrity.py

Checks time-series integrity across reset + first step:
- No duplicate district entries at t=0 (reset then first step overwrites, not appends).
- All building time series report length == current time_step + 1.
"""
import os
import sys
import numpy as np

# Ensure parent repo root on sys.path for local import
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from citylearn.citylearn import CityLearnEnv


def assert_len_ts(env: CityLearnEnv):
    t = env.time_step
    for b in env.buildings:
        # Each of these slices internally to [:t+1]
        assert len(b.net_electricity_consumption) == t + 1
        assert len(b.cooling_electricity_consumption) == t + 1
        assert len(b.heating_electricity_consumption) == t + 1
        assert len(b.dhw_electricity_consumption) == t + 1
        assert len(b.non_shiftable_load_electricity_consumption) == t + 1
        assert len(b.electrical_storage_electricity_consumption) == t + 1
        assert len(b.chargers_electricity_consumption) == t + 1
        assert len(b.washing_machines_electricity_consumption) == t + 1


def main():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'
    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=4)

    # After reset, district has exactly 1 entry (t=0)
    obs, _ = env.reset()
    assert env.time_step == 0
    assert len(env.net_electricity_consumption) == 1
    assert len(env.net_electricity_consumption_cost) == 1
    assert len(env.net_electricity_consumption_emission) == 1
    assert_len_ts(env)

    # First step with zeros: still one district entry (overwritten at t=0), then advance to t=1
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    obs, r, term, trunc, _ = env.step(zeros)
    assert env.time_step == 1
    assert len(env.net_electricity_consumption) == 1, 'duplicate at t=0 detected'
    assert len(env.net_electricity_consumption_cost) == 1
    assert len(env.net_electricity_consumption_emission) == 1
    assert_len_ts(env)

    # Second step: we should now have two district entries (t=0, t=1)
    obs, r, term, trunc, _ = env.step(zeros)
    assert env.time_step == 2
    assert len(env.net_electricity_consumption) == 2
    assert len(env.net_electricity_consumption_cost) == 2
    assert len(env.net_electricity_consumption_emission) == 2
    assert_len_ts(env)

    print('OK - time-series integrity checks passed (no duplicates, correct lengths).')


if __name__ == '__main__':
    main()

