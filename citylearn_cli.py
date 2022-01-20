import argparse
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from agents.marlisa import MARLISA
from agents.sac import SAC
from citylearn import CityLearn, RBC_Agent
from reward_function import reward_function_ma

class CityLearn_CLI:
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def run(self):
        self.__set_run_params()
        start = time.time()
        
        try:
            self.__successful = False
            self.__start_timestamp = datetime.now()
            self.__runner()
            self.__logger.debug(f'Cost - {self.__env.cost()}, Simulation time (min) - {(time.time()-start)/60.0}')
            self.__successful = True
        finally:
            self.__end_timestamp = datetime.now()
            self.__save()

    def __set_run_params(self):
        agent_name = self.kwargs['agent_name']
        reward_style = self.kwargs['reward_style']
        self.__simulation_id = self.kwargs.get('simulation_id') if self.kwargs.get('simulation_id') is not None else f'{agent_name}-{reward_style}'
        self.__output_directory = self.kwargs.get('output_directory',None)

        if self.__output_directory is not None:
            os.makedirs(self.__output_directory,exist_ok=True)
        else:
            self.__output_directory = ''

        self.set_logger()
        self.__set_env()
        self.__step_kwargs = {
            'style':self.kwargs['reward_style'],
            'previous_electricity_demand':None,
            'previous_carbon_intensity':None,
            'exponential_scaling_factor':0.002,
        }
        self.__set_agent()

    def __set_env(self):
        arg_spec = inspect.getfullargspec(CityLearn)
        kwargs = {
            key:value for (key, value) in self.kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
        }
        self.__env = CityLearn(**kwargs)

    def __set_agent(self):
        observations_spaces, actions_spaces = self.__env.get_state_action_spaces()
        agent_handlers = self.get_agent_handlers()
        constructor_kwargs = {
            'building_ids':self.__env.building_ids,
            'buildings_states_actions':self.__env.buildings_states_actions_filename, 
            'building_info':self.__env.get_building_information(),
            'observation_spaces':observations_spaces, 
            'action_spaces':actions_spaces,
            **self.kwargs
        }
        agent_name = self.kwargs['agent_name']
        arg_spec = inspect.getfullargspec(agent_handlers[agent_name]['constructor'])
        constructor_kwargs = {
            key:value for (key, value) in constructor_kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
        }
        params = {**agent_handlers[agent_name],**{'constructor_kwargs':constructor_kwargs}}
        self.__agent = params['constructor'](**params['constructor_kwargs'])
        self.__runner = params['runner']

    def get_agent_handlers(self):
        return {
            'marlisa': {
                'constructor':MARLISA,
                'runner':self.__run_marlisa
            },
            'sac':{
                'constructor':SAC,
                'runner':self.__run_marlisa
            },
            'rbc':{
                'constructor':RBC_Agent,
                'runner':self.__run_rbc
            },
        }

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
            'simulation_period':{'default':[0,8760],'nargs':2,'type':int}, 
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
            'seed':{'default':0,'type':int},
        }

    def __run_marlisa(self):
        episode_count = self.kwargs['episode_count']
        deterministic_period_start = self.kwargs['deterministic_period_start']
        select_action_kwarg_keys = ['states','deterministic']
        select_action_kwarg_keys = [
            key for key in select_action_kwarg_keys 
            if key in inspect.getfullargspec(self.__agent.select_action).args
        ]

        for i in range(episode_count): 
            state = self.__env.reset()
            done = False
            j = 0
            is_evaluating = False
            select_action_kwargs = {'states':state,'deterministic':is_evaluating}
            select_action_kwargs = {key:value for key,value in select_action_kwargs.items() if key in select_action_kwarg_keys}
            action, coordination_vars = self.__agent.select_action(**select_action_kwargs)    
            
            while not done:
                self.__logger.debug(f'Episode: {i+1}/{int(episode_count)} | Timestep: {j+1}/{int(self.__env.simulation_period[1])}')
                next_state, reward, done, _ = self.__env.step(action,**self.__step_kwargs)
                self.__step_kwargs['previous_electricity_demand'] = self.__env.buildings_net_electricity_demand
                self.__step_kwargs['previous_carbon_intensity'] = self.__env.current_carbon_intensity
                self.__logger.debug(f'reward: {reward}')
                select_action_kwargs = {'states':next_state,'deterministic':is_evaluating}
                select_action_kwargs = {key:value for key,value in select_action_kwargs.items() if key in select_action_kwarg_keys}
                action_next, coordination_vars_next = self.__agent.select_action(**select_action_kwargs)
                self.__logger.debug(f'action_next: {action_next}')
                self.__agent.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
                coordination_vars = coordination_vars_next
                state = next_state
                action = action_next
                is_evaluating = (j >= deterministic_period_start)
                j += 1

    def __run_rbc(self):
        episode_count = self.kwargs['episode_count']

        for i in range(episode_count): 
            _ = self.__env.reset()
            done = False
            j = 0
            
            while not done:
                self.__logger.debug(f'Episode: {i+1}/{int(episode_count)} | Timestep: {j+1}/{int(self.__env.simulation_period[1])}')
                hour = list(self.__env.buildings.values())[0].sim_results['hour'][self.__env.time_step]
                action = self.__agent.select_action([hour])
                _, _, done, _ = self.__env.step(action,**self.__step_kwargs)
                j += 1

    def set_logger(self):
        filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.log')
        logging.basicConfig(filename=filepath,format='%(asctime)s %(message)s',filemode='w') 
        logger = logging.getLogger() 
        logger.setLevel(logging.DEBUG)
        self.__logger = logger

    def __save(self):
        data = {
            'metadata':{
                'start_timestamp':self.__start_timestamp,
                'end_timestamp':self.__end_timestamp,
                'successful':self.__successful,
            },
            'env':self.__env,
            'agents':self.__agent
        }
        filepath = os.path.join(self.__output_directory,f'{self.__simulation_id}.pkl')
        directory = '/'.join(filepath.split('/')[0:-1])
        
        if directory != filepath:
            os.makedirs(directory,exist_ok=True)
        else:
            pass

        with open(filepath,'wb') as f:
            pickle.dump(data,f)

def main():
    parser = argparse.ArgumentParser(prog='reward_function_exploration',formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Explore different reward functions in CityLearn environment.')
    parser.add_argument('agent_name',type=str,choices=list(CityLearn_CLI().get_agent_handlers().keys()),help='Simulation agent.')
    parser.add_argument('reward_style',type=str,choices=reward_function_ma.get_styles(),help='Reward function style.')
    parser.add_argument('-id','--simulation_id',type=str,dest='simulation_id',help='ID used to name simulation output files. The default is <agent_name>-<reward_style>.')
    parser.add_argument('-d','--output_directory',type=str,dest='output_directory',help='Directory to store simulation environment, agent and log.')
    parser.add_argument('-e','--episode_count',type=int,default=1,dest='episode_count',help='Number of episodes.')
    parser.add_argument('-dps','--deterministic_period_start',type=int,default=int(7500),dest='deterministic_period_start',help='Deterministic period start index.')

    # env kwargs
    for env_kwarg, arg_kwargs in CityLearn_CLI.get_env_params().items():
        parser.add_argument(f'--{env_kwarg}',dest=env_kwarg,**arg_kwargs,help=' ')

    # agent kwargs
    for agent_kwarg, arg_kwargs in CityLearn_CLI.get_agent_params().items():
        parser.add_argument(f'--{agent_kwarg}',dest=agent_kwarg,**arg_kwargs,help=' ')

    args = parser.parse_args()
    kwargs = {key:value for (key, value) in args._get_kwargs()}

    # run simulation
    cli = CityLearn_CLI(**kwargs)
    cli.run()

if __name__ == '__main__':
    sys.exit(main())