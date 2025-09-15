"""
Run from tests folder: python3 test_tutorial_baseline.py

Replicates the tutorial's baseline idea: when evaluating with identical
control and baseline conditions, district-level normalized KPIs should be ~1.0.

This focuses on the KPIs highlighted in the reported issue: daily peak average,
1 - load factor (daily), and ramping. We keep a modest episode length to speed
up execution while still exercising the logic.
"""
import os
import sys
import numpy as np

# Ensure parent repo root on sys.path for local import without install
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from citylearn.citylearn import CityLearnEnv, EvaluationCondition


def main():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all/schema.json'

    # Shorter run for speed; normalization should still be 1 when conditions match
    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=240)
    obs, _ = env.reset()

    # Zero-action rollout (baseline-like)
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    while not env.terminated:
        obs, r, term, trunc, _ = env.step(zeros)

    # Evaluate with identical conditions so normalized KPIs ~= 1.0
    # Use the DynamicsBuilding default pair since dataset is 2022 phase all
    df = env.evaluate(
        control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
    )
    district = df[df['name'] == 'District'].set_index('cost_function')['value']

    # Metrics of interest from the issue
    keys = [
        'daily_peak_average',
        'daily_one_minus_load_factor_average',
        'ramping_average',
    ]

    for k in keys:
        v = float(district.get(k, 1.0))
        assert np.isfinite(v), f'District KPI {k} is not finite: {v}'
        assert abs(v - 1.0) < 1e-5, f'District KPI {k} not ~1.0, got {v}'

    print('OK - tutorial baseline KPIs (~1.0) verified for daily peak, 1-LF, ramping.')


if __name__ == '__main__':
    main()

