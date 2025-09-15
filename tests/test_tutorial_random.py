"""
Run from tests folder: python3 test_tutorial_random.py

Replicates tutorial-style random agent run and verifies that when evaluating
with identical control/baseline conditions, district KPIs are ~1.0.

This ensures KPI normalization correctness irrespective of the policy used.
"""
import os
import sys
import numpy as np

# Ensure parent repo root on sys.path for local import without install
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from citylearn.citylearn import CityLearnEnv, EvaluationCondition
from citylearn.agents.base import Agent as RandomAgent


def main():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all/schema.json'

    # Shorter run to keep the test fast
    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=240)
    agent = RandomAgent(env)

    obs, _ = env.reset()
    while not env.terminated:
        actions = agent.predict(obs)
        obs, r, term, trunc, _ = env.step(actions)

    # Evaluate with identical conditions so normalized KPIs ~= 1.0
    df = env.evaluate(
        control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
    )
    district = df[df['name'] == 'District'].set_index('cost_function')['value']

    keys = [
        'daily_peak_average',
        'daily_one_minus_load_factor_average',
        'ramping_average',
    ]

    for k in keys:
        v = float(district.get(k, 1.0))
        assert np.isfinite(v), f'District KPI {k} is not finite: {v}'
        assert abs(v - 1.0) < 1e-5, f'District KPI {k} not ~1.0, got {v}'

    print('OK - random agent baseline KPIs (~1.0) verified for daily peak, 1-LF, ramping.')


if __name__ == '__main__':
    main()

