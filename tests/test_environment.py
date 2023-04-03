import pickle
import time
import sys
sys.path.insert(0, '..')
from citylearn.citylearn import CityLearnEnv

RESULT_FILEPATH = 'test_environment.pkl'
schema = 'citylearn_challenge_2020_climate_zone_1'

def main():
    # simulation
    env = CityLearnEnv(schema)
    model = env.load_agent()
    model.learn(episodes=1, deterministic_finish=False)

    print(env.evaluate())

    # save simulator to file
    with open(RESULT_FILEPATH, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = time.time() - start_time
    print('runtime:',runtime)