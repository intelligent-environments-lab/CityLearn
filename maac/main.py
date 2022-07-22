import argparse
import torch
import numpy as np
from torch.autograd import Variable
from algo.attention_sac import AttentionSAC
from utils.make_env import make_env
from utils.encoder import encode


def make_parallel_env(env_id, climate_zone):
    """
    To make a CityLearn environment.
    Inputs:
    :param climate_zone:
    :param env_id:
    Outputs:
    :return:
    """
    print(f"The id of the environment is {env_id}; \n")
    env = make_env(climate_zone)
    return env


def run(config):
    """
    The main loop.
    Input:
    :param config: TODO
    Output:
    :return: TODO
    """
    # 1. Initialize E parallel environments with N agents, including # 2. replay buffer
    # => env = make_parallel_env(some parameters)
    env = make_parallel_env(config.env_id, config.climate_zone)

    encoder, state_dim = encode(env)

    model = AttentionSAC.init_from_env(env, state_dim, config.buffer_length)

    # 3. FOR i_ep = 1 ... num_episode DO:
    for ep_i in range(0, config.n_episodes):
        # 4. T_update <- 0
        t = 0
        explore = True

        if ep_i % 10000 == 0:
            print("Episodes %i of %i" % (ep_i + 1, config.n_episodes))

        # 5. Reset environments, and get initial o_{i}^{e} for each agent, i
        # => model.prep_rollouts()
        obs = env.reset()

        # 6. FOR t = 1...steps per episode DO:
        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.hstack(obs[i])),
                                  requires_grad=False)
                         for i in range(model.num_agents)]

        # 7. Select actions a_{i}^{e} ~ pi_{i}(·|o_{i}^{e}) for each agent, i, in each environment, e
            agent_actions = model.step(torch_obs, encoder, explore=explore)

        # 8. Send actions to all parallel environments and get o'_{i}^{e}, r'_{i}^{e} for all agents
            next_obs, rewards, dones, _ = env.step(agent_actions)

        # 9. Store transitions for all environments in D
            model.add_to_buffer(encoder, obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs

        # 10. T_update
            t += 1
            explore = t < config.exploration

        # 11. IF T_update >= min steps per update THEN:
        # => normalize the replay buffer
            if model.replay_buffer_length() >= config.batch_size and (t % config.update_every) == 0:
                # TODO: gpu
                model.norm_buffer()

            # 12. FOR j = 1...num critic updates DO:
                for u_i in range(config.num_updates):
                    # 13. Sample minibatch, B
                    samples = model.sample(config.batch_size)

                    # 14 -- 20. UpdateCritic(B) ...
                    for a in model.agents:
                        assert(a.norm_flag == 1)
                    model.update_critics(samples)
                    model.update_policies(samples, soft=True)
                    model.update_all_targets()
            # EndFor
    # 21. T_update <- 0
    # EndIf
    # EndFor
    # EndFor

    print("run")
    print(env.cost())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimenting Actor-Attention-Critic")
    parser.add_argument("--env_id", default="1", help="Name of environment")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--climate_zone", default=5, type=int)
    parser.add_argument("--episode_length", default=8760*2, type=int)
    parser.add_argument("--batch_size",
                        default=128, type=int,
                        help="Batch size for training")
    parser.add_argument("--update_every", default=50, type=int)
    parser.add_argument("--num_updates", default=40, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--exploration", default=3000, type=int)

    config = parser.parse_args()

    run(config)
