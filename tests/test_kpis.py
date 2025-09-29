"""
Run from tests folder: python3 test_kpis.py

Ensures parent repo root is on sys.path so local 'citylearn' package is importable
without installing. Alternative: run from repo root using `python -m tests.test_kpis`.

Checks KPI normalization behavior:
- With control_condition == baseline_condition, district-level KPIs should be ~1.
"""
import os
import sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import numpy as np

from citylearn.citylearn import CityLearnEnv, EvaluationCondition


def main():
    schema = '../data/datasets/citylearn_challenge_2022_phase_all/schema.json'
    # Short run for speed
    env = CityLearnEnv(schema, central_agent=True, episode_time_steps=48)
    obs, _ = env.reset()

    # zero actions rollout
    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    while not env.terminated:
        obs, r, term, trunc, _ = env.step(zeros)

    # Evaluate with same control/baseline condition => KPIs ~ 1
    df = env.evaluate(
        control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
    )
    # Only check normalized district KPIs (not raw discomfort, etc.)
    normalized_keys = [
        'ramping_average',
        'daily_one_minus_load_factor_average',
        'monthly_one_minus_load_factor_average',
        'daily_peak_average',
        'all_time_peak_average',
    ]
    district = df[(df['level'] == 'district') & (df['cost_function'].isin(normalized_keys))]
    vals = district['value'].to_numpy(dtype=float)
    # Allow tiny numerical tolerance
    assert np.all(np.isfinite(vals)), f'District KPI contains non-finite values: {district.to_string(index=False)}'
    assert np.all(np.abs(vals - 1.0) < 1e-5), f'District KPI values not ~1.0: {district.to_string(index=False)}'
    print('OK - KPI normalization ~1.0 for identical control/baseline conditions.')


if __name__ == '__main__':
    main()
