import pickle
from citylearn.citylearn import CityLearnEnv
from citylearn.simulator import Simulator
from citylearn.utilities import read_json

RESULT_FILEPATH = 'simulation.pkl'
SCHEMA_FILEPATH = '/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/citylearn/CityLearn/data/cc2022_d1/schema.json'

def main():
    schema = read_json(SCHEMA_FILEPATH)
    citylearn_env = CityLearnEnv(schema)
    agents = citylearn_env.load_agents()
    simulator = Simulator(citylearn_env, agents, schema['episodes'])
    simulator.simulate()

    with open(RESULT_FILEPATH,'wb') as f:
        pickle.dump(simulator,f)

if __name__ == '__main__':
    main()