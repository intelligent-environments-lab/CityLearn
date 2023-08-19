import sys
sys.path.insert(0, '..')
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

# Initialize environment
dataset_name = 'baeda_3dem'
env = CityLearnEnv(dataset_name, buildings=[0], central_agent=True)

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
model = SAC('MlpPolicy', env, verbose=2, learning_starts=env.time_steps, seed=0)
model.learn(total_timesteps=env.time_steps*3, log_interval=1)

# Evaluation
observations = env.reset()

while not env.done:
    actions, _ = model.predict(observations, deterministic=True)
    observations, reward, _, _ = env.step(actions)

print(env.evaluate_citylearn_challenge())