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

def run(reward_style,simulation_filepath=None,log_filepath=None):
    directory = '/'.join(log_filepath.split('/')[0:-1])
    
    if directory != log_filepath:
        os.makedirs(directory,exist_ok=True)
    else:
        pass

    logging.basicConfig(filename=log_filepath,format='%(asctime)s %(message)s',filemode='w') 
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    # Load environment
    climate_zone = 5
    building_ids = ["Building_"+str(i) for i in [1]]
    params_env = {
        'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0, 8760*4-1), 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False
    }
    # We will use 1 episode if we intend to simulate a real-time RL controller (like in the CityLearn Challenge)
    # In climate zone 5, 1 episode contains 5 years of data, or 8760*5 time-steps.
    n_episodes = 1
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

    for e in range(n_episodes): 
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
                'exponential_scaling_factor':0.01,
            }
            next_state, reward, done, misc = env.step(action,**step_kwargs)
            logger.debug(f'previous_electricity_demand: {previous_electricity_demand}, previous_carbon_intensity: {previous_carbon_intensity}')
            logger.debug(f'next_state: {next_state}, reward: {reward}, done: {done}, misc: {misc}')
            previous_electricity_demand = env.buildings_net_electricity_demand
            previous_carbon_intensity = env.current_carbon_intensity
            action_next, coordination_vars_next = agents.select_action(next_state, deterministic=is_evaluating)
            logger.debug(f'action_next: {action_next}, coordination_vars_next: {coordination_vars_next}')
            agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
            coordination_vars = coordination_vars_next
            state = next_state
            action = action_next
            is_evaluating = (j > 3*8760)
            j += 1
            
    logger.debug(f'Reward style: {reward_style}, Loss - {env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')
    data = {'env':env,'agents':agents}
    __save(data,filepath=simulation_filepath)

def __save(data,filepath='citylearn.pkl'):
    directory = '/'.join(filepath.split('/')[0:-1])
    
    if directory != filepath:
        os.makedirs(directory,exist_ok=True)
    else:
        pass

    with open(filepath,'wb') as f:
        pickle.dump(data,f)

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('reward_style',type=str,choices=reward_function_ma(None,None).styles,help='Reward function style.')
    parser.add_argument('-sf','--simulation_filepath',type=str,default='simulation.pkl',dest='simulation_filepath',help='Filepath to write simulation.')
    parser.add_argument('-lf','--log_filepath',type=str,default='std.log',dest='log_filepath',help='Filepath to write log.')
    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(run)
    args_for_func = {k:getattr(args,k,None) for k in arg_spec.args}
    run(**args_for_func)

if __name__ == '__main__':
    sys.exit(main())