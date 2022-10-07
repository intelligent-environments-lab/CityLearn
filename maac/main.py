import argparse
import time
import torch
import numpy as np
from torch.autograd import Variable
from algo.attention_sac import AttentionSAC
from logger.logx import EpochLogger
from logger.plotting import EpisodeStats
from utils.make_env import make_env
from utils.encoder import encode
from utils.misc import count_vars


def make_parallel_env(env_id, climate_zone):
    """
    To make a CityLearn environment.
    Inputs:
    :param climate_zone:
    :param env_id:
    Outputs:
    :return:
    """
    print(f"The id of the environment is {env_id};")
    env = make_env(climate_zone)
    return env


def run(config, logger_kwargs=dict()):
    """
    The main loop.
    Input:
    :param config: configuration of the program setup
    :param logger_kwargs:
    Output:
    :return:
    """

    # Logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_parallel_env(config.env_id, config.climate_zone)

    encoder, state_dim = encode(env)

    model = AttentionSAC.init_from_env(env, state_dim, config.buffer_length)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [model.critic1, model.critic2,
                                                         model.agents[0].policy, model.agents[4].policy])
    logger.log('\nNumber of parameters: \t critic1: %d, \t critic2: %d, \t pi[0]: %d, \t pi[4]: %d\n' % var_counts)

    # Set up model saving
    logger.setup_pytorch_saver(model.critic1)
    logger.setup_pytorch_saver(model.agents[0].policy)

    start_time = time.time()
    t = 0
    n_envs = []
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_rewards=np.zeros(config.n_episodes),
        episode_costs=np.zeros(config.n_episodes)
    )

    explore = True
    deterministic = False

    for ep_i in range(0, config.n_episodes):
        print("Episodes %i of %i" % (ep_i + 1,
                                     config.n_episodes))
        ep_ret = 0
        env = make_parallel_env(config.env_id, config.climate_zone)
        obs = env.reset()

        for et_i in range(config.episode_length):
            if et_i % 200 == 0:
                print("Episode time %i of %i" % (et_i + 1, config.episode_length))
            torch_obs = [Variable(torch.Tensor(np.hstack(obs[i])),
                                  requires_grad=False)
                         for i in range(model.num_agents)]

            # Select actions a_{i}^{e} ~ pi_{i}(·|o_{i}^{e}) for each agent, i, in each environment, e
            agent_actions = model.step(torch_obs, encoder, explore=explore, deterministic=deterministic)

            # Send actions to all parallel environments and get o'_{i}^{e}, r'_{i}^{e} for all agents
            next_obs, rewards, dones, _ = env.step(agent_actions)
            if (et_i + 1) % 50 == 0:
                print(np.array(rewards).mean())
            ep_ret += np.array(rewards).mean()

            model.add_to_buffer(encoder, obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs

            explore = (t <= config.exploration)
            deterministic = (t >= config.episode_length * 0.75)

            t += 1

            if t <= 100:
                assert explore
            elif 101 <= t <= 150:
                assert not deterministic
            elif 151 <= t <= 600:
                assert not explore and deterministic

            # Update statistics
            stats.episode_rewards[ep_i] += np.array(rewards).mean()

            # End of trajectory handling
            if t % config.episode_length == 0 and t > 0:
                stats.episode_costs[ep_i] = env.cost().get("total")
                print("{}th episode reward is: {}".format(ep_i + 1, stats.episode_rewards[ep_i]))
                print("{}th episode total cost is: {}".format(ep_i + 1, stats.episode_costs[ep_i]))
                logger.store(EpRet=ep_ret, EpLen=t)
                ep_ret, ep_len = 0, 0

            if t >= config.update_after and model.replay_buffer_length() >= config.batch_size \
                    and (t % config.update_every) == 0:
                model.norm_buffer()

                for u_i in range(config.num_updates):
                    samples = model.sample(config.batch_size)

                    for a in model.agents:
                        assert (a.pca_flag == 1)
                    model.update_critics(samples)
                    model.update_policies(samples)
                    model.update_all_targets()

        print(f"ran {ep_i + 1}th episode \n")
        n_envs.append(env)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('Time', time.time() - start_time)
    logger.dump_tabular()

    return n_envs, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Actor-Attention-Critic")
    parser.add_argument("--env_id", default="5", help="Name of environment")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--climate_zone", default=5, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--batch_size",
                        default=64, type=int,
                        help="Batch size for training")
    parser.add_argument("--update_after", default=80)
    parser.add_argument("--update_every", default=10, type=int)
    parser.add_argument("--num_updates", default=1, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--exploration", default=100, type=int)
    parser.add_argument("--exp_name", type=str, default="maac")
    parser.add_argument("--seed", type=int, default=42)

    config = parser.parse_args()

    from maac.logger.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(config.exp_name, config.seed)

    run(config, logger_kwargs=logger_kwargs)
