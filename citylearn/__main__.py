import argparse
from datetime import datetime
from citylearn.__init__ import __version__
import inspect
from pathlib import Path
import pickle
import sys
from citylearn.citylearn import CityLearnEnv
from citylearn.simulator import Simulator
from citylearn.utilities import read_json

def simulate(schema: str, result_filepath: str = None):
    citylearn_env = CityLearnEnv(schema)
    agents = citylearn_env.load_agents()
    simulator = Simulator(citylearn_env,agents,schema['episodes'])

    try:
        simulator.simulate()
    
    finally:
        result_filepath = f'simulation_{datetime.utcnow().replace(microsecond=0)}' if result_filepath is None else result_filepath
        result_filepath = '.'.join(result_filepath.split('.')) + '.pkl'

        with open(result_filepath, 'wb') as f:
            pickle.dump(citylearn_env, f)

def main():
    parser = argparse.ArgumentParser(prog='citylearn', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=('''
        An open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement 
        Learning (RL) for building energy coordination and demand response in cities.'''
    ))
    parser.add_argument('--version', action='version', version='%(prog)s' + f' {__version__}')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    subparser_simulate = subparsers.add_parser('simulate',description='Run simulation.')
    subparser_simulate.add_argument('schema',type=Path,help='CityLearn data set name or schema absolute filepath.')
    subparser_simulate.add_argument('-f','--result_filepath',type=Path,dest='result_filepath',help='Filepath to write simulation pickle object to upon completion.')
    subparser_simulate.set_defaults(func=simulate)
    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    args_for_func = {k:getattr(args, k) for k in arg_spec.args}
    args.func(**args_for_func)

if __name__ == '__main__':
    sys.exit(main())