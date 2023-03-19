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
    env = CityLearnEnv(schema)
    agent = env.load_agent()
    simulator = Simulator(env, agent, 2, logging_level=5)
    simulator.simulate()

    print(simulator.env_history)
    print(simulator.env.evaluate())
    assert False

    # save simulator to file
    with open(RESULT_FILEPATH, 'wb') as f:
        pickle.dump(simulator,f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = time.time() - start_time
    print('runtime:',runtime)