from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from citylearn.citylearn_pettingzoo import make_citylearn_env

# Adopted from https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py

if __name__ == "__main__":
    def env_creator(args):
        schema = 'citylearn_challenge_2022_phase_1'
        return PettingZooEnv(make_citylearn_env(schema))

    env = env_creator({})
    register_env("citylearn", env_creator)

    tune.run(
        "SAC",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "citylearn",
            # General
            "num_gpus": 0,
            "framework": 'torch',
            "num_workers": 2,
            # Method specific
            "multiagent": {
                "policies": env.env.agents,
                # "policy_mapping_fn": (lambda agent_id, episode, **kwargs: int(agent_id.split('_')[1])),
                "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),

            },
        },
    )