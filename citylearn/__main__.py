import argparse
from citylearn.__init__ import __version__
import inspect
from pathlib import Path
import pickle
import sys
from citylearn.citylearn import CityLearnEnv
from citylearn.simulator import Simulator

def simulate(schema: str, filepath: str = None, keep_env_history: bool = None, logging_level: int = None):
    env = CityLearnEnv(schema)
    agents = env.load_agent()
    simulator = Simulator(
        env, 
        agents, 
        episodes=env.schema['episodes'], 
        keep_env_history=keep_env_history, 
        logging_level=logging_level
    )

    try:
        simulator.simulate()
    
    finally:
        with open(filepath, 'wb') as f:
            pickle.dump(simulator, f)

def main():
    parser = argparse.ArgumentParser(prog='citylearn', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=('''
        An open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement 
        Learning (RL) for building energy coordination and demand response in cities.'''
    ))
    parser.add_argument('--version', action='version', version='%(prog)s' + f' {__version__}')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    subparser_simulate = subparsers.add_parser('simulate', description='Run simulation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser_simulate.add_argument('schema', type=str, help='CityLearn dataset name or schema path.')
    subparser_simulate.add_argument('-f', '--filepath', type=Path, dest='filepath', default='citylearn_simulator.pkl', help='Filepath to write simulation pickle object to upon completion.')
    subparser_simulate.add_argument('-k', '--keep_env_history', action='store_true', dest='keep_env_history', help='Indicator to store environment state at the end of each episode.')
    subparser_simulate.add_argument("-l", "--logging_level", type=int, default=50, dest='logging_level', help='Logging level where increasing the level silences lower level information.')
    subparser_simulate.set_defaults(func=simulate)
    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    args_for_func = {k:getattr(args, k) for k in arg_spec.args}
    args.func(**args_for_func)

if __name__ == '__main__':
    sys.exit(main())