import gym
# import Chargym_Charging_Station
from citylearn import CityLearn
from pathlib import Path
import argparse
import numpy
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core as core
from spinup.utils.logx import EpochLogger
from Replay_Buffer import ReplayBuffer
from parameters_ddpg import Parameters
import os
import scipy


class ddpg:

    def __init__(self, logger_kwargs=dict()):

        # -------------------- Definition of parameters used for DDPG -------------------------
        [Params, name_of_env, hid, l, epochs, steps_per_epoch, gamma, seed, exp_name, replay_size, max_ep_len, polyak, pi_lr, q_lr,
        batch_size, act_noise, start_steps, update_after, update_every, num_test_episodes,save_freq] = Parameters()
        # -------------------------------------------------------------------------------------

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        # print(locals())
        torch.manual_seed(seed)
        np.random.seed(seed)


        # -------------Folder to save the Results-----------------
        current_folder = os.getcwd() + '\RESULTS'
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)

        # -------------------- Define the environment to use -----------------------
        # name_of_env = 'CityLearn'
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--env", default="CityLearn")
        # args = parser.parse_args()
        # env = gym.make(args.env)
        # self.test_env = gym.make(args.env)

        # Select the climate zone and load environment
        climate_zone = 5
        sim_period = (0, 8760 * 4 - 1)
        params = {'data_path': Path("data/Climate_Zone_" + str(climate_zone)),
                  'building_attributes': 'building_attributes.json',
                  'weather_file': 'weather_data.csv',
                  'solar_profile': 'solar_generation_1kW.csv',
                  'carbon_intensity': 'carbon_intensity.csv',
                  'building_ids': ["Building_" + str(i) for i in [4, 6, 7, 8]],
                  'buildings_states_actions': 'buildings_state_action_space.json',
                  'simulation_period': sim_period,
                  'cost_function': ['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand',
                                    'net_electricity_consumption', 'carbon_emissions'],
                  'central_agent': False,
                  'save_memory': False}

        env = CityLearn(**params)

        observations_spaces, actions_spaces = env.get_state_action_spaces()
        building_info = env.get_building_information()

        self.actions_spaces = actions_spaces
        self.reset_action_tracker()

        # ---------------------------  Initialization --------------------------------------------
        # env, test_env = env(), env()
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = core.MLPActorCritic(env.observation_space, env.action_space)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size))

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        # -------------------------------------------------------------------------------------

        '''=================   Main Loop for the Algorithm  =======================  '''
        # Prepare for interaction with environment
        total_steps = steps_per_epoch * epochs
        start_time = time.time()

        # Retrieve Information from environment
        o, ep_ret, ep_len = env.reset(), [], 0

        # ---------------   Main Loop ----------------------- #
        for t in range(total_steps):
        
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            #while done!==0
            # if t > start_steps:
            a = self.get_action(o)
            # else:
            #     a = env.action_space.sample()

            # Apply the action and retrieve environment information
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    self.update(batch,polyak,gamma)

            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs):
                    self.logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                [mean_test_rew,mean_test_len]=self.test_agent(num_test_episodes,max_ep_len,act_noise,act_dim,act_limit)

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=False)
                self.logger.log_tabular('TestEpRet', with_min_and_max=False)
                #logger.log_tabular('EpLen', average_only=True)
                #logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                #logger.log_tabular('QVals', with_min_and_max=True)
                #logger.log_tabular('LossPi', average_only=True)
                #logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
                # scipy.io.savemat(current_folder + '\Replay_buffer.mat', {'Replay_Buffer': replay_buffer})
                torch.save(self.ac, current_folder + '\ actor' + str(epoch) + '.pt')
                scipy.io.savemat(current_folder + '\Test' + str(epoch) + '.mat',
                {'TestEpRet': mean_test_rew, 'TestEpLen': mean_test_len})

    '''=================   Auxilary Functions  =======================  '''
    def compute_loss_q(self,data,gamma):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    def reset_action_tracker(self):
        self.action_tracker = []

    def compute_loss_pi(self,data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self,data,polyak,gamma):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data,gamma)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


    # def get_action(self,o, noise_scale,act_dim,act_limit):
    #     a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
    #     a += noise_scale * np.random.randn(act_dim)
    def get_action(self, o):
        hour_day = o[0][0]
        multiplier = 0.4
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 7 and hour_day <= 11:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]
        elif hour_day >= 12 and hour_day <= 15:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]
        elif hour_day >= 16 and hour_day <= 18:
            a = [[-0.11 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]
        elif hour_day >= 19 and hour_day <= 22:
            a = [[-0.06 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]

        # Early nightime: store DHW and/or cooling energy
        if hour_day >= 23 and hour_day <= 24:
            a = [[0.085 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]
        elif hour_day >= 1 and hour_day <= 6:
            a = [[0.1383 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in
                 range(len(self.actions_spaces))]

        self.action_tracker.append(a)
        return np.array(a, dtype='object')

        # return np.clip(a, -act_limit, act_limit)

    def test_agent(self,num_test_episodes,max_ep_len,noise_scale,act_dim,act_limit):
        test_rew = []
        test_len = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0,act_dim,act_limit))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            test_rew.append(ep_ret)
            test_len.append(ep_len)
        mean_test_rew = sum(test_rew) / num_test_episodes
        mean_test_len = sum(test_len) / num_test_episodes
        return mean_test_rew, mean_test_len

if __name__ == '__main__':
    seed = 0
    exp_name = 'DDPG_main'
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    ddpg(logger_kwargs=logger_kwargs)
