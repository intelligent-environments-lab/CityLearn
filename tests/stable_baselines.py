import sys
sys.path.insert(0, '..')
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from citylearn.citylearn import CityLearnEnv, StableBaselines3Wrapper

# Initialize environment
dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(dataset_name)

# Wrap for SB3 compatibility
env = StableBaselines3Wrapper(env)

# Perform compatibility check
try:
    check_env(env)
    print('Passed test!! CityLearn is compatible with SB3 when using the StableBaselines3Wrapper.')
finally:
    pass

# Run simulation with SAC policy
model = SAC('MlpPolicy', env, verbose=1, seed=0)
model.learn(total_timesteps=1000, log_interval=4)

# Evaluation
print(env.evaluate())