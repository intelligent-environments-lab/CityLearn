import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from gym import spaces
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log, pymongo_client):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    #args.action_space = env_info["action_space"]
    args.action_spaces = env_info["action_spaces"]
    args.actions_dtype = env_info["actions_dtype"]
    args.normalise_actions = env_info.get("normalise_actions",
                                                False) # if true, action vectors need to sum to one


    # create function scaling agent action tensors to and from range [0,1]
    ttype = th.FloatTensor if not args.use_cuda else th.cuda.FloatTensor
    mult_coef_tensor = ttype(args.n_agents, args.n_actions)
    action_min_tensor = ttype(args.n_agents, args.n_actions)
    if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
        for _aid in range(args.n_agents):
            for _actid in range(args.action_spaces[_aid].shape[0]):
                _action_min = args.action_spaces[_aid].low[_actid]
                _action_max = args.action_spaces[_aid].high[_actid]
                mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
    elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):    # NOTE: This was added to handle scenarios like simple_reference since the action space is Tuple
        for _aid in range(args.n_agents):
            for _actid in range(args.action_spaces[_aid].spaces[0].shape[0]):
                _action_min = args.action_spaces[_aid].spaces[0].low[_actid]
                _action_max = args.action_spaces[_aid].spaces[0].high[_actid]
                mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
            for _actid in range(args.action_spaces[_aid].spaces[1].shape[0]):
                _action_min = args.action_spaces[_aid].spaces[1].low[_actid]
                _action_max = args.action_spaces[_aid].spaces[1].high[_actid]
                tmp_idx = _actid + args.action_spaces[_aid].spaces[0].shape[0]
                mult_coef_tensor[_aid, tmp_idx] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, tmp_idx] = np.asscalar(_action_min)

    args.actions2unit_coef = mult_coef_tensor
    args.actions2unit_coef_cpu = mult_coef_tensor.cpu()
    args.actions2unit_coef_numpy = mult_coef_tensor.cpu().numpy()
    args.actions_min = action_min_tensor
    args.actions_min_cpu = action_min_tensor.cpu()
    args.actions_min_numpy = action_min_tensor.cpu().numpy()

    def actions_to_unit_box(actions):
        if isinstance(actions, np.ndarray):
            return args.actions2unit_coef_numpy * actions + args.actions_min_numpy
        elif actions.is_cuda:
            return args.actions2unit_coef * actions + args.actions_min
        else:
            return args.args.actions2unit_coef_cpu  * actions + args.actions_min_cpu

    def actions_from_unit_box(actions):
        if isinstance(actions, np.ndarray):
            return th.div((actions - args.actions_min_numpy), args.actions2unit_coef_numpy)
        elif actions.is_cuda:
            return th.div((actions - args.actions_min), args.actions2unit_coef)
        else:
            return th.div((actions - args.actions_min_cpu), args.actions2unit_coef_cpu)

    # make conversion functions globally available
    args.actions2unit = actions_to_unit_box
    args.unit2actions = actions_from_unit_box

    action_dtype = th.long if not args.actions_dtype == np.float32 else th.float
    if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
        actions_vshape = 1 if not args.actions_dtype == np.float32 else max([i.shape[0] for i in args.action_spaces])
    elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
        actions_vshape = 1 if not args.actions_dtype == np.float32 else \
                                       max([i.spaces[0].shape[0] + i.spaces[1].shape[0] for i in args.action_spaces])
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (actions_vshape,), "group": "agents", "dtype": action_dtype},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }

    if not args.actions_dtype == np.float32:
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
    else:
        preprocess = {}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1 if args.runner_scope == "episodic" else 2,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = - args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        if getattr(args, "runner_scope", "episodic") == "episodic":
            episode_batch = runner.run(test_mode=False, learner=learner)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size) and (buffer.episodes_in_buffer > getattr(args, "buffer_warmup", 0)):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)
        elif getattr(args, "runner_scope", "episode") == "transition":
            runner.run(test_mode=False,
                       buffer=buffer,
                       learner=learner,
                       episode=episode)
        else:
            raise Exception("Undefined runner scope!")

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            if getattr(args, "testing_on", True):
                for _ in range(n_test_runs):
                    if getattr(args, "runner_scope", "episodic") == "episodic":
                        runner.run(test_mode=True, learner=learner)
                    elif getattr(args, "runner_scope", "episode") == "transition":
                        runner.run(test_mode=True,
                                   buffer = buffer,
                                   learner = learner,
                                   episode = episode)
                    else:
                        raise Exception("Undefined runner scope!")

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # learner.save_models(save_path, args.unique_token, model_save_time)

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config