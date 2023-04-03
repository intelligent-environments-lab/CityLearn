import sys
sys.path.insert(0, '..')
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

# Initialize environment
dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(dataset_name, central_agent=True)
env.buildings = env.buildings[0:1]

# Normalization wrapper
env = NormalizedObservationWrapper(env)

# Wrap for SB3 compatibility
env = StableBaselines3Wrapper(env)

# Perform compatibility check
try:
    check_env(env)
    print('Passed test!! CityLearn is compatible with SB3 when using the StableBaselines3Wrapper.')
finally:
    pass

# Run simulation with SAC policy
model = SAC('MlpPolicy', env, verbose=2, learning_starts=8760, seed=0)
model.learn(total_timesteps=8760*2, log_interval=8760)

# Evaluation
print(env.evaluate())