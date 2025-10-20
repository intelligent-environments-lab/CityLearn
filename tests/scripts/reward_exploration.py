import argparse
import concurrent.futures
from datetime import datetime
import shutil
import inspect
import itertools
import os
from pathlib import Path
import subprocess
import sys
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import SolarPenaltyAndComfortReward
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

def get_combined_data(key, directory):
    data_list = []
    simulation_ids = pd.read_csv(os.path.join(directory, 'reward_exploration_simulation_ids.csv'))

    # environment
    for d in os.listdir(directory):
        if 'simulation' in d and os.path.isdir(os.path.join(directory, d)):
            simulation_id = d
            d = os.path.join(directory, d)
            data = pd.read_csv(os.path.join(d, f'{simulation_id}-{key}.csv'))         
            data_list.append(data)

        else:
            continue

    data = pd.concat(data_list, ignore_index=True, sort=False)
    data = data.merge(simulation_ids, on='simulation_id', how='left')

    return data

def run_work_order(work_order_filepath, max_workers=None, virtual_environment_path=None, windows_system=None):
    work_order_filepath = Path(work_order_filepath)

    if virtual_environment_path is not None:    
        if windows_system:
            virtual_environment_command = f'"{os.path.join(virtual_environment_path, "Scripts", "Activate.ps1")}"'
        else:
            virtual_environment_command = f'source "{os.path.join(virtual_environment_path, "bin", "activate")}"'
    else:
        virtual_environment_command = 'echo "No virtual environment"'

    with open(work_order_filepath,mode='r') as f:
        args = f.read()
    
    args = args.strip('\n').split('\n')
    args = [f'{virtual_environment_command} && {a}' for a in args]
    max_workers = max_workers if max_workers is not None else cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a, 'shell':True}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            except Exception as e:
                print(e)

def simulate(simulation_id, schema, solar_penalty_coefficient, comfort_coefficient, episodes, simulation_output_path, buildings=None):
    # set env
    env = CityLearnEnv(schema, central_agent=True, buildings=buildings)
    env.reward_function = SolarPenaltyAndComfortReward(env, coefficients=[solar_penalty_coefficient, comfort_coefficient])
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    simulation_output_path = os.path.join(simulation_output_path, simulation_id)

    if os.path.isdir(simulation_output_path):
        shutil.rmtree(simulation_output_path)
    else:
        pass

    os.makedirs(simulation_output_path, exist_ok=True)

    # set agent
    callback = SaveDataCallback(env, simulation_id, simulation_output_path, episodes)
    model = SAC(
        'MlpPolicy', 
        env,
        verbose=2, 
        learning_starts=env.unwrapped.time_steps, 
        learning_rate=0.001, 
        gamma=0.9, 
        tau=0.005, 
        batch_size=512, 
        seed=0,
    )

    # train agent
    model.learn(total_timesteps=(env.unwrapped.time_steps - 1)*episodes, log_interval=episodes + 1, callback=callback)

    # test
    observations = env.reset()
    start_timestamp = datetime.utcnow()

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)

    save_data(env, simulation_id, simulation_output_path, start_timestamp, episodes, 'test')

def save_data(env: CityLearnEnv, simulation_id, simulation_output_path, start_timestamp, episode, mode):
    # save runtime summary data
    end_timestamp = datetime.utcnow()
    timer_data = pd.DataFrame([{
        'simulation_id': simulation_id,
        'mode': mode,
        'episode': episode,
        'start_timestamp': start_timestamp, 
        'end_timestamp': end_timestamp
    }])
    timer_filepath = os.path.join(simulation_output_path, f'{simulation_id}-timer.csv')

    if os.path.isfile(timer_filepath):
        existing_data = pd.read_csv(timer_filepath)
        timer_data = pd.concat([existing_data, timer_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    timer_data.to_csv(timer_filepath, index=False)
    del timer_data

    # save environment summary data
    data_list = []

    for i, b in enumerate(env.buildings):
        env_data = pd.DataFrame({
            'net_electricity_consumption': b.net_electricity_consumption,
            'net_electricity_consumption_without_storage': b.net_electricity_consumption_without_storage,
            'net_electricity_consumption_without_storage_and_partial_load': b.net_electricity_consumption_without_storage_and_partial_load,
            'net_electricity_consumption_without_storage_and_partial_load_and_pv': b.net_electricity_consumption_without_storage_and_partial_load_and_pv,
            'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature,
            'indoor_dry_bulb_temperature_without_partial_load': b.indoor_dry_bulb_temperature_without_partial_load,
            'cooling_demand': b.cooling_demand,
            'cooling_demand_without_partial_load': b.cooling_demand_without_partial_load,
            'heating_demand': b.heating_demand,
            'heating_demand_without_partial_load': b.heating_demand_without_partial_load,
            'electrical_storage_soc': b.electrical_storage.soc,
            'dhw_storage_soc': b.dhw_storage.soc,
        })
        env_data['time_step'] = env_data.index
        env_data['mode'] = mode
        env_data['episode'] = episode
        env_data['building_id'] = i
        env_data['building_name'] = b.name
        env_data['simulation_id'] = simulation_id
        data_list.append(env_data)
    
    env_filepath = os.path.join(simulation_output_path, f'{simulation_id}-environment.csv')

    if os.path.isfile(env_filepath):
        existing_data = pd.read_csv(env_filepath)
        data_list = [existing_data] + data_list
        del existing_data
    else:
        pass
    
    env_data = pd.concat(data_list, ignore_index=True, sort=False)
    env_data.to_csv(env_filepath, index=False)
    del data_list
    del env_data
    
    # save reward data
    reward_data = pd.DataFrame(env.rewards, columns=['reward'])
    reward_data['time_step'] = reward_data.index
    reward_data['building_name'] = None
    reward_data['mode'] = mode
    reward_data['episode'] = episode
    reward_data['simulation_id'] = simulation_id
    reward_filepath = os.path.join(simulation_output_path, f'{simulation_id}-reward.csv')

    if os.path.isfile(reward_filepath):
        existing_data = pd.read_csv(reward_filepath)
        reward_data = pd.concat([existing_data, reward_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    reward_data.to_csv(reward_filepath, index=False)
    del reward_data

    # save KPIs
    ## building level
    kpi_data = env.evaluate()
    kpi_data['mode'] = mode
    kpi_data['episode'] = episode
    kpi_data['simulation_id'] = simulation_id
    kpi_filepath = os.path.join(simulation_output_path, f'{simulation_id}-kpi.csv')

    if os.path.isfile(kpi_filepath):
        existing_data = pd.read_csv(kpi_filepath)
        kpi_data = pd.concat([existing_data, kpi_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    kpi_data.to_csv(kpi_filepath, index=False)
    del kpi_data

def set_work_order(schema, buildings, coefficient_start, coefficient_end, coefficient_step, episodes, simulation_output_path=None, filepath=None):
    coefficient_list = np.arange(coefficient_start, coefficient_end, step=coefficient_step)
    coefficient_list = itertools.product(coefficient_list, coefficient_list)
    coefficient_list = [list(c) for c in coefficient_list]
    work_order = ''
    simulation_output_path = 'reward_exploration_simulation_output' if simulation_output_path is None else simulation_output_path
    os.makedirs(simulation_output_path, exist_ok=True)
    
    for i, c in enumerate(coefficient_list):
        work_order += f'python reward_exploration.py simulate simulation_{i} {schema} {c[0]} {c[1]} {episodes} {simulation_output_path} -b {" ".join([str(b) for b in buildings])}\n'

    filepath = 'reward_exploration.sh' if filepath is None else filepath

    with open(filepath, 'w') as f:
        f.write(work_order)

    simulation_ids = pd.DataFrame(coefficient_list, columns=['solar_penalty_coefficient', 'comfort_coefficient'])
    simulation_ids['simulation_id'] = 'simulation_' + simulation_ids.index.astype(str)
    simulation_ids.to_csv(os.path.join(simulation_output_path, 'reward_exploration_simulation_ids.csv'), index=False)

class SaveDataCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, simulation_id, simulation_output_path, episodes, verbose=0):
        super(SaveDataCallback, self).__init__(verbose)
        self.env = env
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.episodes = episodes
        self.episode = 0
        self.start_timestamp = datetime.utcnow()
        self.mode = 'train'

    def _on_step(self) -> bool:
        # save timer data
        if self.env.time_step == self.env.time_steps - 2:
            save_data(
                self.env, 
                self.simulation_id, 
                self.simulation_output_path, 
                self.start_timestamp, 
                self.episode, 
                self.mode
            )
            self.episode += 1
            self.start_timestamp = datetime.utcnow()

        else:
            pass

        return True
    
def main():
    parser = argparse.ArgumentParser(prog='bs2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')

    # set work order
    subparser_set_work_order = subparsers.add_parser('set_work_order')
    subparser_set_work_order.add_argument('schema', type=Path)
    subparser_set_work_order.add_argument('-b', '--buildings', dest='buildings', nargs='+', default=[1], type=int)
    subparser_set_work_order.add_argument('-s', '--coefficient_start', dest='coefficient_start', default=0.0, type=float)
    subparser_set_work_order.add_argument('-e', '--coefficient_end', dest='coefficient_end', default=2.1, type=float)
    subparser_set_work_order.add_argument('-t', '--coefficient_step', dest='coefficient_step', default=0.1, type=float)
    subparser_set_work_order.add_argument('-p', '--episodes', dest='episodes', default=10, type=int)
    subparser_set_work_order.add_argument('-r', '--simulation_output_path', dest='simulation_output_path', default=Path('reward_exploration_simulation_output'), type=Path)
    subparser_set_work_order.add_argument('-f', '--filepath', dest='filepath', default=Path('reward_exploration.sh'), type=Path)
    subparser_set_work_order.set_defaults(func=set_work_order)
    
    # simulate
    subparser_simulate = subparsers.add_parser('simulate')
    subparser_simulate.add_argument('simulation_id', type=str)
    subparser_simulate.add_argument('schema', type=str)
    subparser_simulate.add_argument('solar_penalty_coefficient', type=float)
    subparser_simulate.add_argument('comfort_coefficient', type=float)
    subparser_simulate.add_argument('episodes', type=int)
    subparser_simulate.add_argument('simulation_output_path', type=Path)
    subparser_simulate.add_argument('-b', '--buildings', dest='buildings', nargs='+', default=[1], type=int)
    subparser_simulate.set_defaults(func=simulate)

    # run work order
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.add_argument('work_order_filepath', type=Path)
    subparser_run_work_order.add_argument('-m', '--max_workers', dest='max_workers', default=4, type=int)
    subparser_run_work_order.set_defaults(func=run_work_order)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())