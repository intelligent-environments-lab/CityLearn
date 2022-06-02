import pickle
from citylearn.simulator import Simulator
from citylearn.utilities import read_json

RESULT_FILEPATH = 'simulation.pkl'
SCHEMA_FILEPATH = 'data/cc2022_d1/schema.json'

def main():
    schema = read_json(SCHEMA_FILEPATH)
    citylearn_env, agents, episodes = Simulator.load(schema)
    simulator = Simulator(citylearn_env, agents, episodes)
    simulator.simulate()

    with open(RESULT_FILEPATH,'wb') as f:
        pickle.dump(simulator,f)

if __name__ == '__main__':
    main()