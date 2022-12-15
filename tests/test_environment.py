import pickle
import time
import sys
sys.path.insert(0, '..')
from citylearn.citylearn import CityLearnEnv
from citylearn.simulator import Simulator

RESULT_FILEPATH = 'test_environment.pkl'
schema = 'citylearn_challenge_2020_climate_zone_1'

def main():
    # simulation
    citylearn_env = CityLearnEnv(schema)
    agent = citylearn_env.load_agent()
    simulator = Simulator(citylearn_env, agent, citylearn_env.schema['episodes'])
    simulator.simulate()

    # save simulator to file
    with open(RESULT_FILEPATH, 'wb') as f:
        pickle.dump(simulator,f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = time.time() - start_time
    print('runtime:',runtime)