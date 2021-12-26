import argparse
import inspect
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from agents.marlisa import MARLISA
from agents.rl_agents_coord import RL_Agents_Coord
from citylearn import CityLearn, RBC_Agent
from reward_function import reward_function_ma

def run(**kwargs):
    agent_name = kwargs['agent_name']
    reward_style = kwargs['reward_style']
    simulation_id = kwargs.get('simulation_id') if kwargs.get('simulation_id') is not None else f'{agent_name}-{reward_style}'
    output_directory = os.path.join(kwargs['data_path'],'reward_function_exploration')
    os.makedirs(output_directory,exist_ok=True)
    
    # get logger
    log_filepath = os.path.join(output_directory,f'{simulation_id}.log')
    logger = get_logger(log_filepath)
    
    # run simulation
    env, agent, step_kwargs, runner = get_run_params(**kwargs)
    kwargs = {**kwargs,'env':env,'agent':agent,'step_kwargs':step_kwargs,'logger':logger}
    start = time.time()
    env, agent = runner(**kwargs)
    logger.debug(f'Cost - {env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')

    # save simulation
    data = {'env':env,'agents':agent}
    simulation_filepath = os.path.join(output_directory,f'{simulation_id}.pkl')
    save(data,simulation_filepath)

def get_run_params(**kwargs):
    env = get_env(**kwargs)
    step_kwargs = {
        'style':kwargs['reward_style'],
        'previous_electricity_demand':None,
        'previous_carbon_intensity':None,
        'exponential_scaling_factor':0.002,
    }
    agent_params = get_agent_params(env,**kwargs)
    agent = agent_params['constructor'](**agent_params['constructor_kwargs'])
    runner = agent_params['runner']
    return env, agent, step_kwargs, runner

def get_env(**kwargs):
    arg_spec = inspect.getfullargspec(CityLearn)
    kwargs = {
        key:value for (key, value) in kwargs.items()
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    env = CityLearn(**kwargs)
    return env

def get_agent_params(env,**kwargs):
    observations_spaces, actions_spaces = env.get_state_action_spaces()
    agent_handlers = __get_agent_handlers()
    
    constructor_kwargs = {
        'building_ids':env.building_ids,
        'buildings_states_actions':env.buildings_states_actions_filename, 
        'building_info':env.get_building_information(),
        'observation_spaces':observations_spaces, 
        'action_spaces':actions_spaces,
        **kwargs
    }
    agent_name = kwargs['agent_name']
    arg_spec = inspect.getfullargspec(agent_handlers[agent_name]['constructor'])
    constructor_kwargs = {
        key:value for (key, value) in constructor_kwargs.items()
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    params = {**agent_handlers[agent_name],**{'constructor_kwargs':constructor_kwargs}}
    return params

def __get_agent_handlers():
    return {
        'rl_agents_coord': {
            'constructor':RL_Agents_Coord,
            'runner':__run_marlisa
        },
        'marlisa': {
            'constructor':MARLISA,
            'runner':__run_marlisa
        },
        'rbc':{
            'constructor':RBC_Agent,
            'runner':__run_rbc
        },
    }

def __get_env_params():
    cost_function_choices = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions']
    default_building_ids = [f'Building_{i}' for i in range(1,10)]
    return {
        'data_path':{'required':True,'type':Path},
        'building_attributes':{'default':Path('building_attributes.json'),'type':Path}, 
        'weather_file':{'default':Path('weather_data.csv'),'type':Path}, 
        'solar_profile':{'default':Path('solar_generation_1kW.csv'),'type':Path}, 
        'carbon_intensity':{'default':Path('carbon_intensity.csv'),'type':Path}, 
        'building_ids':{'default':default_building_ids,'nargs':'+','type':str}, 
        'buildings_states_actions':{'default':'buildings_state_action_space.json','type':str}, 
        'simulation_period':{'default':[0,8760],'nargs':2,'type':int}, 
        'cost_function':{'default':cost_function_choices,'nargs':'+','choices':cost_function_choices,'type':str},
        'central_agent':{'default':False,'action':'store_true'},
        'save_memory':{'default':False,'action':'store_true'},
    }

def __get_agent_params():
    return {
        'hidden_dim':{'default':[256,256],'nargs':2,'type':int}, 
        'discount':{'default':0.99,'type':float}, 
        'tau':{'default':5e-3,'type':float}, 
        'lr':{'default':3e-4,'type':float}, 
        'batch_size':{'default':256,'type':int}, 
        'replay_buffer_capacity':{'default':1e5,'type':float}, 
        'regression_buffer_capacity':{'default':8760,'type':float}, 
        'start_training':{'default':600,'type':int}, 
        'exploration_period':{'default':7500,'type':int},
        'start_regression':{'default':500,'type':int}, 
        'information_sharing':{'default':False,'action':'store_true'}, 
        'pca_compression':{'default':0.95,'type':float}, 
        'action_scaling_coef':{'default':1.0,'type':float}, 
        'reward_scaling':{'default':0.5,'type':float}, 
        'update_per_step':{'default':1,'type':int}, 
        'iterations_as':{'default':2,'type':int}, 
        'safe_exploration':{'default':False,'action':'store_true'},
        'seed':{'default':0,'type':int},
    }

def __run_marlisa(**kwargs):
    env = kwargs['env']
    agent = kwargs['agent']
    logger = kwargs['logger']
    step_kwargs = kwargs['step_kwargs']
    episode_count = kwargs['episode_count']
    deterministic_period_start = kwargs['deterministic_period_start']

    for i in range(episode_count): 
        state = env.reset()
        done = False
        j = 0
        is_evaluating = False
        action, coordination_vars = agent.select_action(state, deterministic=is_evaluating)    
        
        while not done:
            logger.debug(f'Episode: {i+1}/{int(episode_count)} | Timestep: {j+1}/{int(env.simulation_period[1])}')
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
    episode_count = kwargs['episode_count']

    for i in range(episode_count): 
        _ = env.reset()
        done = False
        j = 0
        
        while not done:
            logger.debug(f'Episode: {i+1}/{int(episode_count)} | Timestep: {j+1}/{int(env.simulation_period[1])}')
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

def save(data,filepath):
    directory = '/'.join(filepath.split('/')[0:-1])
    
    if directory != filepath:
        os.makedirs(directory,exist_ok=True)
    else:
        pass

    with open(filepath,'wb') as f:
        pickle.dump(data,f)

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('agent_name',type=str,choices=list(__get_agent_handlers().keys()),help='Simulation agent.')
    parser.add_argument('reward_style',type=str,choices=reward_function_ma.get_styles(),help='Reward function style.')
    parser.add_argument('-id','--simulation_id',type=str,dest='simulation_id',help='ID used to name simulation output files. The default is <agent_name>-<reward_style>.')
    parser.add_argument('-e','--episode_count',type=int,default=1,dest='episode_count',help='Number of episodes.')
    parser.add_argument('-dps','--deterministic_period_start',type=int,default=int(7500),dest='deterministic_period_start',help='Deterministic period start index.')
    # env kwargs
    for env_kwarg, arg_kwargs in __get_env_params().items():
        parser.add_argument(f'--{env_kwarg}',dest=env_kwarg,**arg_kwargs,help=' ')

    # agent kwargs
    for agent_kwarg, arg_kwargs in __get_agent_params().items():
        parser.add_argument(f'--{agent_kwarg}',dest=agent_kwarg,**arg_kwargs,help=' ')

    args = parser.parse_args()
    kwargs = {key:value for (key, value) in args._get_kwargs()}
    run(**kwargs)

if __name__ == '__main__':
    sys.exit(main())