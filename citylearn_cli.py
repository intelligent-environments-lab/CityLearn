import argparse
import concurrent.futures
from copy import deepcopy
from datetime import datetime
import inspect
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import pickle
import time
import subprocess
import sys
from agents.marlisa import MARLISA
from agents.rbc import BasicRBC, OptimizedRBC
from agents.sac import SAC
from citylearn import CityLearn
from database import CityLearnDatabase
from reward_function import reward_function_ma
from utilities import get_data_from_path, write_json

# module logger
LOGGER = logging.getLogger(__name__)

class CityLearn_CLI:
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.__set_output_directory()
        self.__set_simulation_id()
        self.__set_logger()

    def run(self):
        try:
            LOGGER.debug(f'Started simulation.')
            LOGGER.debug(f'kwargs: {self.kwargs}.')
            self.__set_run_params()
            start = time.time()
            self.__successful = False
            self.__start_timestamp = datetime.now()
            self.__end_timestamp = None
            self.__write_progress()
            self.__run()
            self.__successful = True
            LOGGER.debug(f'Cost - {self.__env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')
        
        except (Exception, KeyboardInterrupt) as e:
            LOGGER.exception(e,stack_info=True)

            if self.__timestep >= 0:
                self.__write_end_timestep = self.__timestep + 1
                self.__database_timestep_update()
            else:
                pass
        
        finally:
            LOGGER.debug(f'Ending simulation ...')
            self.__end_timestamp = datetime.now()
            self.__write_progress()
            self.__write_pickle()
            self.__close_database()
            LOGGER.debug(f'Ended simulation.')

    def __set_run_params(self):
        self.__set_env()
        self.__step_kwargs = {
            'style':self.kwargs['reward_style'],
            'previous_electricity_demand':None,
            'previous_carbon_intensity':None,
            'exponential_scaling_factor':0.002,
        }
        self.__is_rbc = self.kwargs['agent_name'] in ['basic_rbc','optimized_rbc']
        self.__set_agent()
        self.__initialize_database()
        self.__update_write_timestep(reset=True)
        self.__episode = -1
        self.__timestep = -1
        self.__episode_actions = []
        self.__episode_rewards = []

    def __set_env(self):
        arg_spec = inspect.getfullargspec(CityLearn)
        kwargs = {
            key:value for (key, value) in self.kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
        }
        self.__env = CityLearn(**kwargs)

    def __set_agent(self):
        observations_spaces, actions_spaces = self.__env.get_state_action_spaces()
        agent_name = self.kwargs['agent_name']
        constructor = self.get_agent_constructors()[agent_name]
        constructor_kwargs = {
            'building_ids':self.__env.building_ids,
            'buildings_states_actions':self.__env.buildings_states_actions_filename, 
            'building_info':self.__env.get_building_information(),
            'observation_spaces':observations_spaces, 
            'action_spaces':actions_spaces,
            **self.kwargs
        }
        arg_spec = inspect.getfullargspec(constructor)
        constructor_kwargs = {
            key:value for (key, value) in constructor_kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
        }

        if self.__is_rbc:
            constructor_kwargs['actions_spaces'] = actions_spaces
        else:
            pass

        self.__agent = constructor(**constructor_kwargs)

    @staticmethod
    def get_agent_constructors():
        return {
            'marlisa':MARLISA,
            'sac':SAC,
            'basic_rbc':BasicRBC,
            'optimized_rbc':OptimizedRBC,
        }

    def __run(self):
        select_action_kwarg_keys = ['states','deterministic']
        select_action_kwarg_keys = [
            key for key in select_action_kwarg_keys 
            if key in inspect.getfullargspec(self.__agent.select_action).args
        ]
        episode_count = self.kwargs['episode_count']

        while self.__episode < episode_count - 1:
            self.__episode += 1
            self.__timestep = -1
            j = 0
            self.__update_write_timestep(reset=True)
            self.__episode_actions = []
            self.__episode_rewards = []
            done = False
            is_evaluating = False
            
            if self.__is_rbc:
                hour_day = list(self.__env.buildings.values())[0].sim_results['hour'][self.__env.time_step]
                action = self.__agent.select_action(hour_day)
            else:
                state = self.__env.reset()
                select_action_kwargs = {'states':state,'deterministic':is_evaluating}
                select_action_kwargs = {key:value for key,value in select_action_kwargs.items() if key in select_action_kwarg_keys}
                action, coordination_vars = self.__agent.select_action(**select_action_kwargs)
            
            while not done:
                while j < self.__write_end_timestep and not done:
                    LOGGER.debug(f'Episode: {self.__episode+1}/{episode_count} | Timestep: {j+1}/{int(self.__env.simulation_period[1])}')
                    next_state, reward, done, _ = self.__env.step(action,**self.__step_kwargs)
                    self.__timestep = j
                    self.__episode_actions.append(action)
                    self.__episode_rewards.append(reward)
                    self.__step_kwargs['previous_electricity_demand'] = self.__env.buildings_net_electricity_demand
                    self.__step_kwargs['previous_carbon_intensity'] = self.__env.current_carbon_intensity
                
                    if self.__is_rbc:
                        hour_day = list(self.__env.buildings.values())[0].sim_results['hour'][self.__env.time_step]
                        action = self.__agent.select_action(hour_day)

                    else:
                        select_action_kwargs = {'states':next_state,'deterministic':is_evaluating}
                        select_action_kwargs = {key:value for key,value in select_action_kwargs.items() if key in select_action_kwarg_keys}
                        action_next, coordination_vars_next = self.__agent.select_action(**select_action_kwargs)
                        self.__agent.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
                        coordination_vars = coordination_vars_next
                        state = next_state
                        action = action_next
                        is_evaluating = (j >= self.kwargs['deterministic_period_start'])
                    
                    j += 1

                self.__database_timestep_update()
                self.__write_progress()
                self.__update_write_timestep(reset=False)

    def __set_logger(self):
        filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.log')
        LOGGER.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filepath,mode='w')
        formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(process)d - %(thread)d: %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        
    def __close_database(self):
        if self.kwargs['write_sqlite']:
            kwargs = {'successful':self.__successful}
            self.__database.end_simulation(**kwargs)
            LOGGER.debug(f'Closed database.')
        else:
            pass

    def __initialize_database(self):
        if self.kwargs['write_sqlite']:
            database_filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.db')
            self.__database = CityLearnDatabase(database_filepath,self.__env,self.__agent,overwrite=True,apply_changes=True)
            kwargs = deepcopy(self.kwargs)
            kwargs['simulation_name'] = self.__simulation_id
            self.__database.initialize(**kwargs)
            LOGGER.debug(f'Initialized database with filepath - {self.__database.filepath}.')
        else:
            pass

    def __database_timestep_update(self):
        if self.kwargs['write_sqlite']:
            LOGGER.debug(f'Updating database timeseries.')
            kwargs = {
                'start_timestep':self.__write_start_timestep,
                'end_timestep':self.__write_end_timestep,
                'episode':self.__episode,
                'action':self.__episode_actions,
                'reward':self.__episode_rewards
            }
            self.__database.timestep_update(**kwargs)
            LOGGER.debug(f'Finished updating database timeseries.')
        else:
            pass

    def __update_write_timestep(self,reset=False):
        if reset:
            self.__write_start_timestep = 0
        else:
            self.__write_start_timestep = self.__write_end_timestep

        self.__write_end_timestep = min([self.__write_start_timestep + self.kwargs['write_frequency'],self.__env.simulation_period[1]])

    def __write_progress(self):
        data = {
            'simulation_id':self.__simulation_id,
            'pid':os.getpid(),
            'start_timestamp':self.__start_timestamp,
            'end_timestamp':self.__end_timestamp,
            'last_update_timestamp':datetime.now(),
            'last_update_episode':self.__episode + 1,
            'last_update_timestep':self.__timestep,
            'successful':self.__successful,
        }
        filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.json')
        write_json(filepath,data)
        LOGGER.debug(f'Wrote progess to {filepath}.')
    
    def __write_pickle(self):
        if self.kwargs['write_pickle']:
            # data = {
            #     'metadata':{
            #         'start_timestamp':self.__start_timestamp,
            #         'end_timestamp':self.__end_timestamp,
            #         'successful':self.__successful,
            #     },
            #     'env':self.__env,
            #     'agents':self.__agent
            # }
            data = {
                'agents':self.__agent
            }
            filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.pkl')

            with open(filepath,'wb') as f:
                pickle.dump(data,f)
                LOGGER.debug(f'Wrote simulation to pickle file - {filepath}.')

        else:
            pass

    def __set_simulation_id(self):
        agent_name = self.kwargs['agent_name']
        reward_style = self.kwargs['reward_style']
        self.__simulation_id = self.kwargs.get('simulation_id')\
            if self.kwargs.get('simulation_id') is not None else f'{agent_name}-{reward_style}'

    def __set_output_directory(self):
        self.__output_directory = self.kwargs.get('output_directory',None)

        if self.__output_directory is not None:
            os.makedirs(self.__output_directory,exist_ok=True)
        else:
            self.__output_directory = ''

    @staticmethod
    def get_env_params():
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
            'simulation_period':{'default':[0,8759],'nargs':2,'type':int}, 
            'cost_function':{'default':cost_function_choices,'nargs':'+','choices':cost_function_choices,'type':str},
            'central_agent':{'default':False,'action':'store_true'},
            'save_memory':{'default':False,'action':'store_true'},
        }

    @staticmethod
    def get_agent_params():
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
            'basic_rbc':{'default':False,'action':'store_true'},
            'seed':{'default':0,'type':int},
        }

def single_simulation_run(**kwargs):
    cli = CityLearn_CLI(**kwargs)

    try:
        cli.run()
    except Exception as e:
        LOGGER.exception(e,stack_info=True)

def multiple_simulation_run(filepath,**kwargs):
    args = get_data_from_path(filepath).split('\n')

    if kwargs.get('work_directory',None) is not None:
        args = [f'cd {kwargs["work_directory"]} && {a}' for a in args]
    else:
        pass

    max_workers = kwargs['max_workers'] if kwargs.get('max_workers',None) is not None else cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a,'shell':True}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            print(future.result())

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('--write_sqlite',action='store_true',help='Write simulation to SQLite database.')
    parser.add_argument('--write_frequency',type=int,default=1000,help='Timestep frequency for writing simulation to SQLite database.')
    parser.add_argument('--write_pickle',action='store_true',help='Write simulation to pickle file.')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # single simulation run
    single_run_subparser = subparsers.add_parser('single',description='Run a single CityLearn simulation on a single process.')
    single_run_subparser.set_defaults(func=single_simulation_run)
    single_run_subparser.add_argument('agent_name',type=str,choices=list(CityLearn_CLI.get_agent_constructors().keys()),help='Simulation agent.')
    single_run_subparser.add_argument('reward_style',type=str,choices=reward_function_ma.get_styles(),help='Reward function style.')
    single_run_subparser.add_argument('-id','--simulation_id',type=str,dest='simulation_id',help='ID used to name simulation output files. The default is <agent_name>-<reward_style>.')
    single_run_subparser.add_argument('-d','--output_directory',type=str,dest='output_directory',help='Directory to store simulation environment, agent and log.')
    single_run_subparser.add_argument('-e','--episode_count',type=int,default=1,dest='episode_count',help='Number of episodes.')
    single_run_subparser.add_argument('-dps','--deterministic_period_start',type=int,default=int(7500),dest='deterministic_period_start',help='Deterministic period start index.')
    # env kwargs
    for env_kwarg, arg_kwargs in CityLearn_CLI.get_env_params().items():
        single_run_subparser.add_argument(f'--{env_kwarg}',dest=env_kwarg,**arg_kwargs,help=' ')
    # agent kwargs
    for agent_kwarg, arg_kwargs in CityLearn_CLI.get_agent_params().items():
        single_run_subparser.add_argument(f'--{agent_kwarg}',dest=agent_kwarg,**arg_kwargs,help=' ')

    # multiple simulation run
    multiple_run_subparser = subparsers.add_parser('multiple',description='Run multiple single CityLearn simulations on a multiple processes.')
    multiple_run_subparser.set_defaults(func=multiple_simulation_run)
    multiple_run_subparser.add_argument('filepath',type=str,help='Filepath to script containing multiple simulation commands.')
    multiple_run_subparser.add_argument('-w','--max_workers',type=int,default=None,dest='max_workers',
        help='Number of processors on the machine to use.'\
            ' Defaults to the number of processors on machine. On Windows, max_workers must be less than or equal to 61.'\
                ' If max_workers is None, then the default chosen will be at most 61, even if more processors are available.'
    )
    multiple_run_subparser.add_argument('-d','--work_directory',type=str,default=None,dest='work_directory',help='Directory to execute runs from.')

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {
        key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())