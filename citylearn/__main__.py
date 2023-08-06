import argparse
from citylearn.__init__ import __version__
import inspect
from pathlib import Path
import pickle
import sys
from citylearn.citylearn import CityLearnEnv

def __learn(**kwargs):
    env = CityLearnEnv(kwargs['schema'])
    model = env.load_agent()
    
    try:
        model.learn(
            episodes=kwargs.get('episodes', env.schema['episodes']),
            deterministic_finish=kwargs['deterministic_finish'],
            logging_level=kwargs['logging_level']
        )
    
    finally:
        with open(kwargs['filepath'], 'wb') as f:
            pickle.dump(model, f)

def main():
    parser = argparse.ArgumentParser(prog='citylearn', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=('''
        An open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement 
        Learning (RL) for building energy coordination and demand response in cities.'''
    ))
    parser.add_argument('--version', action='version', version='%(prog)s' + f' {__version__}')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    subparser_simulate = subparsers.add_parser('learn', description='Run simulation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser_simulate.add_argument('schema', type=str, help='CityLearn dataset name or schema path.')
    subparser_simulate.add_argument('-e', '--episodes', type=int, dest='episodes', default=None, help='Number of training episodes.')
    subparser_simulate.add_argument('-f', '--filepath', type=Path, dest='filepath', default='citylearn_learning.pkl', help='Filepath to write model pickle object to upon completion.')
    subparser_simulate.add_argument('-n', '--deterministic_finish', action='store_true', dest='deterministic_finish', help='Take deterministic actions in the final episode.')
    subparser_simulate.add_argument("-l", "--logging_level", type=int, default=50, dest='logging_level', help='Logging level where increasing the level silences lower level information.')
    subparser_simulate.set_defaults(func=__learn)
    
    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())