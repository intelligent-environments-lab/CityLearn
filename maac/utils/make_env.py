from citylearn import CityLearn
from pathlib import Path
import os

HOURS_PER_YEAR = 8760


def make_env(climate_zone):
    """
        Create a CityLearn environment
        :return:
        """
    params = {'data_path': Path(
        "/Users/xiejiahan/PycharmProjects/CityLearn/data/Climate_Zone_" + str(climate_zone)),
        'building_attributes': 'building_attributes.json',
        'weather_file': 'weather_data.csv',
        'solar_profile': 'solar_generation_1kW.csv',
        'carbon_intensity': 'carbon_intensity.csv',
        'building_ids': ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        'buildings_states_actions': '/Users/xiejiahan/PycharmProjects/CityLearn/buildings_state_action_space'
                                    '.json',
        'simulation_period': (0, 200),
        'cost_function': ['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand',
                          'net_electricity_consumption', 'carbon_emissions'],
        'central_agent': False,
        'save_memory': False}
    env = CityLearn(**params)
    return env


if __name__ == '__main__':
    print(os.getcwd())
    env = make_env(5)
    observations_spaces, actions_spaces = env.get_state_action_spaces()
    print(len(actions_spaces))
