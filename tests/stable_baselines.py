import sys
sys.path.insert(0, '..')
from stable_baselines3.common.env_checker import check_env
from citylearn.citylearn import CityLearnEnvStableBaselines3Wrapper

dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnvStableBaselines3Wrapper(dataset_name, central_agent=True)

try:
    check_env(env)
    print('Passed test!! CityLearn is compatible with SB3 when using the CityLearnEnvStableBaselines3Wrapper.')
except Exception as e:
    print(e)