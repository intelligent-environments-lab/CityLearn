import pickle
import time
from citylearn.citylearn import CityLearnEnv
from citylearn.simulator import Simulator
from citylearn.utilities import read_json

RESULT_FILEPATH = 'simulation.pkl'
schema = 'citylearn_challenge_2022_phase_1'
# schema = read_json('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/citylearn/CityLearn/citylearn/data/citylearn_challenge_2022_phase_1/schema.json')
# schema['episodes'] = 3
# schema['simulation_end_time_step'] = 100
# schema['root_directory'] = '/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/citylearn/CityLearn/citylearn/data/citylearn_challenge_2022_phase_1/'

def main():
    citylearn_env = CityLearnEnv(schema)
    agents = citylearn_env.load_agents()
    simulator = Simulator(citylearn_env, agents, citylearn_env.schema['episodes'])
    simulator.simulate()

    with open(RESULT_FILEPATH,'wb') as f:
        pickle.dump(simulator,f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = time.time() - start_time
    print('runtime:',runtime)