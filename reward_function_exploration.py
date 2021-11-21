import argparse
import inspect
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from agents.marlisa import MARLISA
from citylearn import  CityLearn
from reward_function import reward_function_ma

def run(**kwargs):
    climate_zone = kwargs['climate_zone']
    reward_style = kwargs['reward_style']
    simulation_period_start = kwargs.get('simulation_period_start',0)
    simulation_period_end = kwargs.get('simulation_period_end',8759)
    episode_count = kwargs.get('episode_count',1)
    deterministic_period_start = kwargs.get('deterministic_period_start',3*8760 + 1)
    data_directory = 'data_reward_function_exploration'
    climate_zone_directory = os.path.join(data_directory,f'Climate_Zone_{climate_zone}')
    output_directory = os.path.join(climate_zone_directory,'reward_function_exploration')
    os.makedirs(output_directory,exist_ok=True)
    
    # set logger
    log_filepath = os.path.join(output_directory,f'{reward_style}.log')
    logging.basicConfig(filename=log_filepath,format='%(asctime)s %(message)s',filemode='w') 
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    # Load environment
    building_ids = ["Building_"+str(i) for i in range(1,10)]
    params_env = {
        'data_path':Path(climate_zone_directory), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (simulation_period_start,simulation_period_end), 
        'cost_function': [
            'ramping',
            '1-load_factor',
            'average_daily_peak',
            'peak_demand',
            'net_electricity_consumption',
            'carbon_emissions'
        ], 
        'central_agent': False,
        'save_memory': False
    }
    # Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
    # Can be obtained using observations_spaces[i].low or .high
    env = CityLearn(**params_env)
    observations_spaces, actions_spaces = env.get_state_action_spaces()

    # Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
    building_info = env.get_building_information()
    params_agent = {
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'building_info':building_info,
        'observation_spaces':observations_spaces, 
        'action_spaces':actions_spaces, 
        'hidden_dim':[256,256], 
        'discount':0.99, 
        'tau':5e-3, 
        'lr':3e-4, 
        'batch_size':256, 
        'replay_buffer_capacity':1e5, 
        'regression_buffer_capacity':3e4, 
        'start_training':600, # Start updating actor-critic networks
        'exploration_period':7500, # Just taking random actions
        'start_regression':500, # Start training the regression model
        'information_sharing':True, # If True -> set the appropriate 'reward_function_ma' in reward_function.py
        'pca_compression':.95, 
        'action_scaling_coef':0.5, # Actions are multiplied by this factor to prevent too aggressive actions
        'reward_scaling':5., # Rewards are normalized and multiplied by this factor
        'update_per_step':2, # How many times the actor-critic networks are updated every hourly time-step
        'iterations_as':2,# Iterations of the iterative action selection (see MARLISA paper for more info)
        'safe_exploration':True
    } 

    # Instantiating the control agent(s)
    agents = MARLISA(**params_agent)
    start = time.time()
    previous_electricity_demand = None
    previous_carbon_intensity = None

    for _ in range(episode_count): 
        state = env.reset()
        done = False
        j = 0
        is_evaluating = False
        action, coordination_vars = agents.select_action(state, deterministic=is_evaluating)    
        
        while not done:
            logger.debug(f'Timestep: {j+1}/{int(params_env["simulation_period"][1])}')
            step_kwargs = {
                'style':reward_style,
                'previous_electricity_demand':previous_electricity_demand,
                'previous_carbon_intensity':previous_carbon_intensity,
                'exponential_scaling_factor':0.002,
            }
            next_state, reward, done, misc = env.step(action,**step_kwargs)
            logger.debug(f'previous_electricity_demand: {previous_electricity_demand}')
            logger.debug(f'next_state: {next_state}, reward: {reward}')
            previous_electricity_demand = env.buildings_net_electricity_demand
            previous_carbon_intensity = env.current_carbon_intensity
            action_next, coordination_vars_next = agents.select_action(next_state, deterministic=is_evaluating)
            logger.debug(f'action_next: {action_next}, coordination_vars_next: {coordination_vars_next}')
            agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
            coordination_vars = coordination_vars_next
            state = next_state
            action = action_next
            is_evaluating = (j >= deterministic_period_start)
            j += 1
            
    logger.debug(f'Reward style: {reward_style}, Loss - {env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')
    data = {'env':env,'agents':agents}
    simulation_filepath = os.path.join(output_directory,f'{reward_style}.pkl')
    __save(data,filepath=simulation_filepath)

def __save(data,filepath='citylearn.pkl'):
    directory = '/'.join(filepath.split('/')[0:-1])
    
    if directory != filepath:
        os.makedirs(directory,exist_ok=True)
    else:
        pass

    with open(filepath,'wb') as f:
        pickle.dump(data,f)

def __get_climate_zones():
    climate_zones = []

    for climate_zone_directory in os.listdir('data'):
        if climate_zone_directory.startswith('Climate_Zone'):
            climate_zones.append(climate_zone_directory.split('_')[-1])
        else:
            continue
    
    return climate_zones

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('climate_zone',type=str,choices=__get_climate_zones(),help='Simulation climate zone.')
    parser.add_argument('reward_style',type=str,choices=reward_function_ma.get_styles(),help='Reward function style.')
    parser.add_argument('-e','--episode_count',type=int,default=1,dest='episode_count',help='Number of episodes.')
    parser.add_argument('-sps','--simulation_period_start',type=int,default=0,dest='simulation_period_start',help='Simulation start index.')
    parser.add_argument('-spe','--simulation_period_end',type=int,default=8759,dest='simulation_period_end',help='Simulation end index.')
    parser.add_argument('-dps','--deterministic_period_start',type=int,default=int(8760*3 + 1),dest='deterministic_period_start',help='Deterministic period start index.')
    args = parser.parse_args()
    kwargs = {key:value for (key, value) in args._get_kwargs()}
    run(**kwargs)

if __name__ == '__main__':
    sys.exit(main())