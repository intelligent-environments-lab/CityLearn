import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.vdnstate import VDNState
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam


class CQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.named_params = dict(mac.named_parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1: # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == 'vdn-s':
                self.mixer = VDNState(args)
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.named_params.update(dict(self.mixer.named_parameters()))
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.optimiser = RMSprop(params=self.params,
                                     lr=args.lr,
                                     alpha=args.optim_alpha,
                                     eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.optimiser = Adam(params=self.params,
                                  lr=args.lr,
                                  eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, 0].unsqueeze(-1)
        terminated = batch["terminated"][:, 0].float().unsqueeze(-1)
        mask = 1 - terminated

        t = 0
        chosen_action_qvals, _ = self.mac.forward(batch,
                                                  actions=batch["actions"][:, t:t + 1].detach(),
                                                  t=t)

        t = 1
        best_target_action = self.target_mac.select_actions(batch,
                                                            t_ep=t,
                                                            t_env=None,
                                                            test_mode=True)
        target_max_qvals, _ = self.target_mac.forward(batch, t=t, actions=best_target_action.detach())

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            chosen_action_qvals = chosen_action_qvals.squeeze(-2)
            target_max_qvals = target_max_qvals.squeeze(-2)
            rewards = rewards.squeeze(-2)
            terminated = terminated.squeeze(-2)

        # Calculate 1-step Q-Learning targets
        targets = rewards.expand_as(target_max_qvals) + self.args.gamma * (1 -
                                                        terminated.expand_as(target_max_qvals)) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # Normal L2 loss, take mean over actual data
        assert self.args.runner_scope == "transition", "Runner scope HAS to be transition!"
        loss = (td_error ** 2).mean()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau = getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception("unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("weight_norm", (th.sum(th.cat([th.sum(p**2).unsqueeze(0) for p in self.params]))**0.5).item(), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (batch.batch_size * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (batch.batch_size * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated target network (soft update tau={})".format(tau))

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))