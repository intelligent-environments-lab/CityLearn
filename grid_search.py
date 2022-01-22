from copy import copy, deepcopy
import itertools
import os
import random
import shutil
import pandas as pd
from utilities import read_json, write_data, write_json

SOURCE_DATA_DIRECTORY = 'data/'
DESTINATION_DATA_DIRECTORY = 'grid_search_data/'
SOURCE_CARBON_INTENSITY_FILEPATH = os.path.join(SOURCE_DATA_DIRECTORY,'Climate_Zone_5/carbon_intensity.csv')
SIMULATION_OUTPUT_DIRECTORY = os.path.join(DESTINATION_DATA_DIRECTORY,'simulation_output')
GRID_SEARCH_FILEPATH = 'grid_search.json'
DEFAULT_BUILDING_COUNT = 9
BUILDING_COUNT_MULTIPLIERS = [1,5,10]
WRITE_FREQUENCY = 4000
PYTHON_EXECUTION = f'python -m citylearn_cli --write_sqlite --write_frequency {WRITE_FREQUENCY} single'
TACC_LAUNCHER_JOB_FILEPATH = 'tacc_launcher_job'
CLIMATE_ZONES = [2]
SIMULATION_COUNT = 3

def main():
    if os.path.isdir(DESTINATION_DATA_DIRECTORY):
        shutil.rmtree(DESTINATION_DATA_DIRECTORY)
    else:
        pass

    set_building_data()
    set_grid()

def set_grid():
    grid_search = read_json(GRID_SEARCH_FILEPATH)
    param_names = list(grid_search.keys())
    param_values = list(grid_search.values())
    param_values_grid = list(itertools.product(*param_values))
    grid = pd.DataFrame(param_values_grid,columns=param_names)
    grid['--exploration_period'] = grid['--start_training'] + 1
    grid['--start_regression'] = (grid['--start_training']*0.5).astype(int)
    grid_list = []
    
    for data_path in [os.path.join(DESTINATION_DATA_DIRECTORY,f'Climate_Zone_{i}') for i in CLIMATE_ZONES]:
        for multiplier in BUILDING_COUNT_MULTIPLIERS:
            for i in range(SIMULATION_COUNT):
                grid_copy = grid.copy()
                grid_copy['--data_path'] = data_path
                grid_copy['--building_ids'] = ' '.join([f'Building_{i}' for i in range(1,int(DEFAULT_BUILDING_COUNT*multiplier)+1)])
                grid_copy['--buildings_states_actions'] = os.path.join(data_path,'buildings_state_action_space.json')
                grid_copy['--seed'] = i
                grid_list.append(grid_copy)

    grid = pd.concat(grid_list,ignore_index=True)
    grid['--output_directory'] = SIMULATION_OUTPUT_DIRECTORY
    grid.loc[grid['agent_name']=='sac','reward_style'] = 'sac'
    grid = grid.drop_duplicates()
    grid = grid.sort_values([
        '--data_path',
        '--seed',
        'reward_style',
        'agent_name',
        '--building_ids',
    ])
    grid['--simulation_id'] = grid.reset_index().index.map(lambda x: f'simulation_{x + 1}')
    grid.to_csv(f'{GRID_SEARCH_FILEPATH.split(".")[0]}.csv',index=False)
    script = [
        PYTHON_EXECUTION + ' ' + ' '.join([f'{key if key.startswith("--") else ""} {value}'.strip() for key, value in record.items() if value is not None])
        for record in grid.to_dict(orient='records')
    ]
    script = '\n'.join(script)
    write_data(script,TACC_LAUNCHER_JOB_FILEPATH)
    print('Number of simulations to run:',grid.shape[0])

def set_building_data():
    for source_climate_zone_directory in [os.path.join(SOURCE_DATA_DIRECTORY,d) for d in os.listdir(SOURCE_DATA_DIRECTORY) if d.startswith('Climate_')]:
        destination_climate_zone_directory = os.path.join(DESTINATION_DATA_DIRECTORY,source_climate_zone_directory.split('/')[-1])
        os.makedirs(destination_climate_zone_directory,exist_ok=True)

        # simulation results
        source_simulation_results_filepaths = [os.path.join(source_climate_zone_directory,f) for f in os.listdir(source_climate_zone_directory) if f.startswith('Building')]
        source_simulation_results_filepaths.sort()
        source_simulation_results_filepaths = [source_simulation_results_filepaths for _ in range(int(max(BUILDING_COUNT_MULTIPLIERS)))]
        source_simulation_results_filepaths = [f for l in source_simulation_results_filepaths for f in l]
        destination_simulation_results_filepaths = [os.path.join(destination_climate_zone_directory,f'Building_{i+1}.csv') for i in range(len(source_simulation_results_filepaths))]
        for source, destination in zip(source_simulation_results_filepaths,destination_simulation_results_filepaths): shutil.copy(source,destination)

        # weather and solar generation
        source_weather_filepath = os.path.join(source_climate_zone_directory,'weather_data.csv')
        destination_weather_filepath = os.path.join(destination_climate_zone_directory,'weather_data.csv')
        source_solar_generation_filepath = os.path.join(source_climate_zone_directory,'solar_generation_1kW.csv')
        destination_solar_generation_filepath = os.path.join(destination_climate_zone_directory,'solar_generation_1kW.csv')
        shutil.copy(source_weather_filepath,destination_weather_filepath)
        shutil.copy(source_solar_generation_filepath,destination_solar_generation_filepath)

        # attributes and state, action space
        source_attributes_filepath = os.path.join(source_climate_zone_directory,'building_attributes.json')
        source_state_action_space_filepath = os.path.join('buildings_state_action_space.json')
        destination_attributes_filepath = os.path.join(destination_climate_zone_directory,'building_attributes.json')
        destination_state_action_space_filepath = os.path.join(destination_climate_zone_directory,'buildings_state_action_space.json')
        source_attributes = read_json(source_attributes_filepath)
        source_state_action_space = read_json(source_state_action_space_filepath)
        destination_attributes = {}
        destination_state_action_space = {}

        for building_id in source_attributes:
            building_number = int(building_id.split('_')[-1])
            destination_attributes[building_id] = deepcopy(source_attributes[building_id])
            destination_state_action_space[building_id] = source_state_action_space[building_id]

            for i in range(1,int(max(BUILDING_COUNT_MULTIPLIERS))):
                j = int(len(source_attributes)*i + building_number)
                building_copy_id = f'Building_{j}'
                destination_attributes[building_copy_id] = deepcopy(destination_attributes[building_id])
                destination_attributes[building_copy_id]['File_Name'] = f'{building_copy_id}.csv'
                destination_attributes[building_copy_id] = get_attributes(destination_attributes[building_copy_id],random_seed=j)
                destination_state_action_space[building_copy_id] = deepcopy(destination_state_action_space[building_id])
                destination_state_action_space[building_copy_id] = get_state_action_space(destination_state_action_space[building_copy_id],destination_attributes[building_copy_id])

        write_json(destination_attributes_filepath,destination_attributes,sort_keys=True)
        write_json(destination_state_action_space_filepath,destination_state_action_space,sort_keys=True)

    # set carbon intensity data
    carbon_intensity = pd.read_csv(SOURCE_CARBON_INTENSITY_FILEPATH)
    carbon_intensity['Datetime'] = pd.to_datetime(carbon_intensity['Datetime'])
    carbon_intensity['month'] = carbon_intensity['Datetime'].dt.month
    carbon_intensity['day'] = carbon_intensity['Datetime'].dt.day
    carbon_intensity['hour'] = carbon_intensity['Datetime'].dt.hour
    carbon_intensity = carbon_intensity.groupby(['month','day','hour'])[['kg_CO2/kWh']].mean()
    carbon_intensity = carbon_intensity.reset_index()
    # remove leap year 2/29
    ixs = carbon_intensity[(carbon_intensity['month']==2)&(carbon_intensity['day']==29)].index
    carbon_intensity = carbon_intensity.drop(ixs)

    for climate_zone in [1,2,3,4]:
        destination_filepath = os.path.join(DESTINATION_DATA_DIRECTORY,f'Climate_Zone_{climate_zone}/carbon_intensity.csv')
        carbon_intensity.to_csv(destination_filepath,index=False)

    climate_zone_5_destination_filepath = os.path.join(DESTINATION_DATA_DIRECTORY,'Climate_Zone_5/carbon_intensity.csv')
    shutil.copy(SOURCE_CARBON_INTENSITY_FILEPATH,climate_zone_5_destination_filepath)

def get_state_action_space(state_action_space,attributes):
    # states
    state_action_space['states']['solar_gen'] = True if attributes['Solar_Power_Installed(kW)'] > 0 else False
    state_action_space['states']['cooling_storage_soc'] = True if attributes['Chilled_Water_Tank']['capacity'] > 0 else False
    state_action_space['states']['dhw_storage_soc'] = True if attributes['DHW_Tank']['capacity'] > 0 else False
    state_action_space['states']['electrical_storage_soc'] = True if attributes['Battery']['capacity'] > 0 else False
    # actions
    state_action_space['actions']['cooling_storage'] = True if attributes['Chilled_Water_Tank']['capacity'] > 0 else False
    state_action_space['actions']['dhw_storage'] = True if attributes['DHW_Tank']['capacity'] > 0 else False
    state_action_space['actions']['electrical_storage'] = True if attributes['Battery']['capacity'] > 0 else False

    return state_action_space

def get_attributes(attributes,random_seed=None):
    # randomize applicable values
    if random_seed is not None:
        random.seed(random_seed)
        attributes['Solar_Power_Installed(kW)'] = attributes['Solar_Power_Installed(kW)']*random.randint(0,5)
        attributes['Battery']['capacity'] = attributes['Battery']['capacity']*random.randint(0,2)
        attributes['Heat_Pump']['technical_efficiency'] = random.uniform(0.2,0.3)
        attributes['Heat_Pump']['t_target_heating'] = random.randint(47,50)
        attributes['Heat_Pump']['t_target_cooling'] = random.randint(7,10)
        attributes['Electric_Water_Heater']['efficiency'] = random.uniform(0.9,1.0)
        attributes['Chilled_Water_Tank']['loss_coefficient'] = random.uniform(0.002,0.01)
        attributes['DHW_Tank']['loss_coefficient'] = random.uniform(0.002,0.01)
    else:
        pass

    return attributes

main()