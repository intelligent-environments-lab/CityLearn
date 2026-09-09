#!/usr/bin/env python3
"""Build the final 15-minute EC_Ermesinde dataset from the supplied sources.

The source building/weather data is hourly. Energy columns are split equally
between four 15-minute steps; weather and indoor state are held constant within
each original hour. Escalator source files are copied from the corrected archive
and the two atrium series are rebuilt from the validated platform flows so that
the obsolete duplicate train events cannot reappear in the final dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'datasets' / 'EC_Ermesinde'
ESCALATOR_SOURCE = Path('/home/tiago/Downloads/ERolantes_corrigido')
BUILDING_SOURCE = Path('/home/tiago/Downloads/Building_1.csv')
WEATHER_SOURCE = Path('/home/tiago/Downloads/weather.csv')
SCHEMA_SOURCE = Path('/home/tiago/Downloads/schema_baseline.json.json')


def _repeat_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.index.repeat(4)].reset_index(drop=True)


def build_building() -> None:
    frame = pd.read_csv(BUILDING_SOURCE)
    if len(frame) != 8760:
        raise ValueError(f'Expected 8760 hourly building rows, got {len(frame)}.')
    for column in ('dhw_demand', 'solar_generation'):
        if column not in frame:
            frame[column] = 0.0
    result = _repeat_hourly(frame)
    result['minutes'] = np.tile(np.array([0, 15, 30, 45], dtype='int16'), len(frame))
    for column in ('non_shiftable_load', 'cooling_demand', 'heating_demand', 'dhw_demand', 'solar_generation'):
        result[column] = pd.to_numeric(result[column], errors='raise') / 4.0
    order = ['month', 'day_type', 'hour', 'minutes'] + [c for c in result.columns if c not in {'month', 'day_type', 'hour', 'minutes'}]
    result[order].to_csv(TARGET / 'Building_1.csv', index=False, float_format='%.6f')


def build_weather() -> None:
    frame = pd.read_csv(WEATHER_SOURCE)
    if len(frame) != 8760:
        raise ValueError(f'Expected 8760 hourly weather rows, got {len(frame)}.')
    _repeat_hourly(frame).to_csv(TARGET / 'weather.csv', index=False, float_format='%.6f')


def build_pricing() -> None:
    building = pd.read_csv(BUILDING_SOURCE)
    if len(building) != 8760:
        raise ValueError(f'Expected 8760 hourly building rows, got {len(building)}.')
    hour = building['hour'].to_numpy(dtype='int16')
    hourly_price = np.select(
        [hour <= 7, hour <= 17, hour <= 22],
        [0.120, 0.180, 0.260],
        default=0.180,
    )
    price = np.repeat(hourly_price, 4)
    result = pd.DataFrame({'electricity_pricing': price})
    # The source tariff forecasts were 6, 12 and 24 hours ahead. At 15-minute
    # resolution these are 24, 48 and 96 steps ahead, respectively.
    for index, horizon in enumerate((24, 48, 96), start=1):
        result[f'electricity_pricing_predicted_{index}'] = np.roll(price, -horizon)
    result.to_csv(TARGET / 'pricing.csv', index=False, float_format='%.6f')


def _recompute_derived(frame: pd.DataFrame) -> pd.DataFrame:
    frame['passengers_from_trains_15min'] = frame['passengers_from_trains_15min'].clip(lower=0.0)
    frame['background_pedestrians_15min'] = frame['background_pedestrians_15min'].clip(lower=0.0)
    frame['passengers_expected_15min'] = (
        frame['passengers_from_trains_15min'] + frame['background_pedestrians_15min']
    )
    frame['people_detected'] = (frame['passengers_expected_15min'] > 0.5).astype('int8')
    for column in ('arriving_trains', 'departing_trains', 'minutes_to_next_train', 'available'):
        frame[column] = pd.to_numeric(frame[column], errors='raise').clip(lower=0).astype('int32')
    return frame


def build_escalators() -> None:
    source = {}
    for number in range(1, 7):
        filepath = ESCALATOR_SOURCE / f'EscadaRolante_{number}.csv'
        frame = pd.read_csv(filepath)
        if len(frame) != 35040:
            raise ValueError(f'{filepath} must contain 35040 rows.')
        source[number] = frame

    # Correct the lone year-end lead-direction artefact: its next train belongs
    # outside the annual study period and therefore contributes no passengers.
    for number in (3, 5):
        source[number].loc[source[number].index[-1], 'passengers_from_trains_15min'] = 0.0
        source[number] = _recompute_derived(source[number])

    # The atrium is a shared connection to platform groups 2/3 and 4/5.
    # Rebuilding it from the already corrected platform series removes the 1,230
    # stale duplicate events found in the supplied E1/E2 files. E1 is upstream
    # (towards platforms) and E2 downstream (from platforms); independent
    # background pedestrians remain exactly as provided for each atrium direction.
    for atrium, platform_numbers in ((1, (3, 5)), (2, (4, 6))):
        old = source[atrium].copy()
        aggregate = old.copy()
        aggregate['passengers_from_trains_15min'] = sum(
            source[number]['passengers_from_trains_15min'] for number in platform_numbers
        )
        aggregate['arriving_trains'] = sum(source[number]['arriving_trains'] for number in platform_numbers)
        aggregate['departing_trains'] = sum(source[number]['departing_trains'] for number in platform_numbers)
        aggregate['minutes_to_next_train'] = np.minimum.reduce([
            source[number]['minutes_to_next_train'].to_numpy(dtype='int32') for number in platform_numbers
        ])
        source[atrium] = _recompute_derived(aggregate)

    # E1 inherits the lead-direction boundary from E3/E5. Make it explicit.
    source[1].loc[source[1].index[-1], 'passengers_from_trains_15min'] = 0.0
    source[1] = _recompute_derived(source[1])

    for number, frame in source.items():
        if not np.array_equal(frame['time_step'].to_numpy(dtype='int64'), np.arange(35040, dtype='int64')):
            raise ValueError(f'EscadaRolante_{number}: time_step is not contiguous.')
        if not np.allclose(
            frame['passengers_expected_15min'],
            frame['passengers_from_trains_15min'] + frame['background_pedestrians_15min'],
            rtol=1.0e-6, atol=1.0e-6,
        ):
            raise ValueError(f'EscadaRolante_{number}: expected passengers invariant failed.')
        frame.to_csv(TARGET / f'EscadaRolante_{number}.csv', index=False, float_format='%.4f')

    events = pd.read_csv(ESCALATOR_SOURCE / 'ermesinde_eventos_comboios.csv')
    if len(events) != 200 or events.duplicated().any():
        raise ValueError('Corrected train-event table must contain 200 unique rows.')
    events.to_csv(TARGET / 'ermesinde_eventos_comboios.csv', index=False)


def build_schema() -> None:
    filepath = TARGET / 'schema.json'
    schema = json.loads(SCHEMA_SOURCE.read_text(encoding='utf-8'))
    schema['simulation_start_time_step'] = 0
    schema['simulation_end_time_step'] = 35039
    schema['seconds_per_time_step'] = 900.0
    schema['observations']['minutes'] = {'active': True, 'shared_in_central_agent': True}
    escalator_features = (
        'passengers_from_trains_15min',
        'background_pedestrians_15min',
        'passengers_expected_15min',
        'people_detected',
        'passing_trains',
        'minutes_to_next_train',
        'available',
        'state',
        'requested_state',
        'power_kw',
        'service_required',
        'service_met',
        'unserved_passengers_15min',
    )
    for feature in escalator_features:
        schema['observations'][f'escalator_{feature}'] = {
            'active': True,
            'shared_in_central_agent': False,
        }
    schema['actions']['escalator'] = {'active': True}

    escalators = {}
    for number in range(1, 7):
        if number <= 2:
            powers = {'standby_power': 0.08, 'slow_power': 0.42, 'normal_power': 2.10}
        else:
            # E3--E6 each represent two physical escalators. Values supplied by
            # the student were per physical unit, hence the doubled aggregate load.
            powers = {'standby_power': 0.10, 'slow_power': 0.52, 'normal_power': 2.62}
        escalators[f'EscadaRolante_{number}'] = {
            'type': 'citylearn.energy_model.Escalator',
            'simulation': f'EscadaRolante_{number}.csv',
            'attributes': {
                **powers,
                'minimum_state_steps': 1,
                'service_threshold_passengers': 0.5,
            },
        }
    schema['buildings']['Building_1']['escalators'] = escalators
    filepath.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    build_building()
    build_weather()
    build_pricing()
    build_escalators()
    build_schema()


if __name__ == '__main__':
    main()
