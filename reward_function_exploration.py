import argparse
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from agents.marlisa import MARLISA
from citylearn import CityLearn, RBC_Agent
from reward_function import reward_function_ma

def run(**kwargs):
    agent_name = kwargs['agent_name']
    reward_style = kwargs['reward_style']
    simulation_id = f'{agent_name}-{reward_style}'
    climate_zone_directory = os.path.join('data_reward_function_exploration',f'Climate_Zone_{kwargs["climate_zone"]}')
    output_directory = os.path.join(climate_zone_directory,'reward_function_exploration')
    os.makedirs(output_directory,exist_ok=True)
    
    # get logger
    log_filepath = os.path.join(output_directory,f'{simulation_id}.log')
    logger = get_logger(log_filepath)
    
    # run simulation
    env, agent, step_kwargs, runner = get_run_params(**kwargs)
    kwargs = {**kwargs,'env':env,'agent':agent,'step_kwargs':step_kwargs,'logger':logger}
    start = time.time()
    env, agent = runner(**kwargs)
    logger.debug(f'Loss - {env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')

    # save simulation
    data = {'env':env,'agents':agent}
    simulation_filepath = os.path.join(output_directory,f'{simulation_id}.pkl')
    save(data,filepath=simulation_filepath)

def get_run_params(**kwargs):
    env = get_env(**kwargs)
    step_kwargs = {
        'style':kwargs['reward_style'],
        'previous_electricity_demand':None,
        'previous_carbon_intensity':None,
        'exponential_scaling_factor':0.002,
    }
    agent_params = get_agent_params(kwargs['agent_name'],env)
    agent = agent_params['constructor'](**agent_params['constructor_kwargs'])
    runner = agent_params['runner']
    return env, agent, step_kwargs, runner

def get_env(**kwargs):
    climate_zone = kwargs['climate_zone']
    simulation_period_start = kwargs.get('simulation_period_start',0)
    simulation_period_end = kwargs.get('simulation_period_end',8759)
    data_directory = 'data_reward_function_exploration'
    climate_zone_directory = os.path.join(data_directory,f'Climate_Zone_{climate_zone}')
    output_directory = os.path.join(climate_zone_directory,'reward_function_exploration')
    os.makedirs(output_directory,exist_ok=True)
    building_ids = ["Building_"+str(i) for i in range(1,10)]
    env_kwargs = {
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
    env = CityLearn(**env_kwargs)
    return env

def get_agent_params(agent_name,env):
    observations_spaces, actions_spaces = env.get_state_action_spaces()
    building_info = env.get_building_information()
    building_ids = env.building_ids
    agent_params = {
        'marlisa': {
            'constructor_kwargs':{
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
                'iterations_as':10,# Iterations of the iterative action selection (see MARLISA paper for more info)
                'safe_exploration':True
            },
            'constructor':MARLISA,
            'runner':__run_marlisa
        },
        'rbc':{
            'constructor_kwargs':{
                'actions_spaces':actions_spaces
            },
            'constructor':RBC_Agent,
            'runner':__run_rbc
        },
    }
    return agent_params[agent_name]

def __run_marlisa(**kwargs):
    env = kwargs['env']
    agent = kwargs['agent']
    logger = kwargs['logger']
    step_kwargs = kwargs['step_kwargs']
    episode_count = kwargs.get('episode_count',1)
    deterministic_period_start = kwargs.get('deterministic_period_start',3*8760 + 1)

    for _ in range(episode_count): 
        state = env.reset()
        done = False
        j = 0
        is_evaluating = False
        action, coordination_vars = agent.select_action(state, deterministic=is_evaluating)    
        
        while not done:
            logger.debug(f'Timestep: {j+1}/{int(env.simulation_period[1])}')
            next_state, reward, done, _ = env.step(action,**step_kwargs)
            step_kwargs['previous_electricity_demand'] = env.buildings_net_electricity_demand
            step_kwargs['previous_carbon_intensity'] = env.current_carbon_intensity
            logger.debug(f'reward: {reward}')
            action_next, coordination_vars_next = agent.select_action(next_state, deterministic=is_evaluating)
            logger.debug(f'action_next: {action_next}')
            agent.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
            coordination_vars = coordination_vars_next
            state = next_state
            action = action_next
            is_evaluating = (j >= deterministic_period_start)
            j += 1

    return env, agent

def __run_rbc(**kwargs):
    env = kwargs['env']
    agent = kwargs['agent']
    logger = kwargs['logger']
    step_kwargs = kwargs['step_kwargs']
    episode_count = kwargs.get('episode_count',1)

    for _ in range(episode_count): 
        _ = env.reset()
        done = False
        j = 0
        
        while not done:
            logger.debug(f'Timestep: {j+1}/{int(env.simulation_period[1])}')
            hour = list(env.buildings.values())[0].sim_results['hour'][env.time_step]
            action = agent.select_action([hour])
            _, _, done, _ = env.step(action,**step_kwargs)
            j += 1

    return env, agent

def get_logger(filepath):
    logging.basicConfig(filename=filepath,format='%(asctime)s %(message)s',filemode='w') 
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    return logger

def save(data,filepath='citylearn.pkl'):
    directory = '/'.join(filepath.split('/')[0:-1])
    
    if directory != filepath:
        os.makedirs(directory,exist_ok=True)
    else:
        pass

    with open(filepath,'wb') as f:
        pickle.dump(data,f)

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('climate_zone',type=str,choices=['1','2','3','4','5'],help='Simulation climate zone.')
    parser.add_argument('agent_name',type=str,choices=['marlisa','rbc'],help='Simulation agent.')
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