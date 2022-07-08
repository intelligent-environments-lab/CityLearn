from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import supersuit as ss
from citylearn.citylearn_pettingzoo import make_citylearn_env


schema = './citylearn/data/citylearn_challenge_2022_phase_1/schema.json'
citylearn_pettingzoo_env = make_citylearn_env(schema)

env = ss.pettingzoo_env_to_vec_env_v1(citylearn_pettingzoo_env)

model = PPO(CnnPolicy, env, 
            verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, 
            max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)

model.learn(total_timesteps=2000000)