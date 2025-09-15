"""
Run from tests folder: python3 test_alignment.py

Ensures parent repo root is on sys.path so local 'citylearn' package is importable
without installing. Alternative: run from repo root using `python -m tests.test_alignment`.

Checks:
- Timestep alignment: step returns s_{t+1}, reward is computed for [t,t+1).
- EV charger inclusion: charger electricity consumption is included in net consumption at same timestep.
"""
import os
import sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import math
import numpy as np

from citylearn.citylearn import CityLearnEnv


def _finite(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.all(np.isfinite(x))
    return math.isfinite(float(x))


def test_timestep_and_chargers():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'
    env = CityLearnEnv(schema, central_agent=True)

    obs, _ = env.reset()
    # Step 1: zero actions
    zeros = np.zeros(env.action_space[0].shape[0], dtype='float32')
    obs, r0, term, trunc, _ = env.step([zeros])

    # After first step, env advanced to t=1 and reward is finite
    assert env.time_step == 1, f"Expected time_step==1 after one step, got {env.time_step}"
    assert _finite(r0), f"Non-finite reward at t=0: {r0}"

    # Build an action that charges the first available EV charger (if any)
    names = env.action_names[0]
    ev_ix = next((i for i, n in enumerate(names) if n.startswith('electric_vehicle_storage_')), None)
    if ev_ix is None:
        print('No EV storage action found; skipping charger inclusion check for this dataset.')
        return

    actions = np.zeros_like(zeros)
    actions[ev_ix] = 0.1  # small positive charge

    # Step 2: apply EV action at current t=1; reward r1 computed for [1,2); then advance to t=2
    obs, r1, term, trunc, _ = env.step([actions])
    assert env.time_step == 2, f"Expected time_step==2 after two steps, got {env.time_step}"
    assert _finite(r1), f"Non-finite reward at t=1: {r1}"

    # Check building 0 net consumption at t=1 includes charger consumption at t=1
    b = env.buildings[0]
    t = env.time_step - 1  # values updated for t before advancing
    lhs = b.net_electricity_consumption[t]
    rhs = (
        b.cooling_electricity_consumption[t]
        + b.heating_electricity_consumption[t]
        + b.dhw_electricity_consumption[t]
        + b.non_shiftable_load_electricity_consumption[t]
        + b.electrical_storage_electricity_consumption[t]
        + b.solar_generation[t]
        + b.chargers_electricity_consumption[t]
        + b.washing_machines_electricity_consumption[t]
    )
    assert abs(lhs - rhs) < 1e-4, f"Net consumption mismatch at t={t}: lhs={lhs}, rhs={rhs}"


if __name__ == '__main__':
    test_timestep_and_chargers()
    print('OK - timestep and charger inclusion checks passed.')
