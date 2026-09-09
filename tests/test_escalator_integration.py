import numpy as np
import pandas as pd
from pathlib import Path

from citylearn.citylearn import CityLearnEnv


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data' / 'datasets' / 'EC_Ermesinde'
SCHEMA = DATASET / 'schema.json'


def test_ec_ermesinde_escalators_load_and_apply_three_states():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=8,
        render_mode='none',
        offline=True,
    )
    try:
        observations, _ = env.reset(seed=0)
        assert len(env.buildings[0].escalators) == 6
        assert env.action_names == [[f'escalator_EscadaRolante_{i}' for i in range(1, 7)]]
        assert any('passengers_expected_15min' in name for name in env.observation_names[0])

        env.step([0.0] * 6)
        building = env.buildings[0]
        assert np.isclose(building._Building__escalators_electricity_consumption[0], 0.14)
        assert [escalator.state for escalator in building.escalators] == [0] * 6

        env.step([0.5] * 6)
        assert np.isclose(building._Building__escalators_electricity_consumption[1], 0.73)
        assert [escalator.state for escalator in building.escalators] == [1] * 6

        env.step([1.0] * 6)
        assert np.isclose(building._Building__escalators_electricity_consumption[2], 3.67)
        assert [escalator.state for escalator in building.escalators] == [2] * 6

        metrics = env.evaluate_v2(include_business_as_usual=False)
        assert 'building_escalator_service_service_level_ratio' in set(metrics['cost_function'])
        assert 'district_escalator_energy_electricity_consumption_total_kwh' in set(metrics['cost_function'])
    finally:
        env.close()


def test_ec_ermesinde_escalator_dataset_invariants_and_atrium_rebuild():
    frames = {
        number: pd.read_csv(DATASET / f'EscadaRolante_{number}.csv')
        for number in range(1, 7)
    }
    for frame in frames.values():
        assert len(frame) == 35040
        assert np.array_equal(frame['time_step'].to_numpy(), np.arange(35040))
        assert np.allclose(
            frame['passengers_expected_15min'],
            frame['passengers_from_trains_15min'] + frame['background_pedestrians_15min'],
        )

    assert np.allclose(
        frames[1]['passengers_from_trains_15min'],
        frames[3]['passengers_from_trains_15min'] + frames[5]['passengers_from_trains_15min'],
    )
    assert np.allclose(
        frames[2]['passengers_from_trains_15min'],
        frames[4]['passengers_from_trains_15min'] + frames[6]['passengers_from_trains_15min'],
    )
    assert frames[1]['passengers_from_trains_15min'].iloc[-1] == 0.0
    assert frames[3]['passengers_from_trains_15min'].iloc[-1] == 0.0
    assert frames[5]['passengers_from_trains_15min'].iloc[-1] == 0.0
