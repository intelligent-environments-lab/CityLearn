import numpy as np
print('\n Defining system parameters of ddpg...')

def Parameters():
    name_of_env='CityLearn'
    hid=256
    l=2
    epochs=100
    steps_per_epoch = 4000
    save_freq = 1
    gamma=0.99
    seed=0
    exp_name='ddpg'
    replay_size=1e6
    max_ep_len=1000
    polyak = 0.995
    pi_lr = 1e-3
    q_lr = 1e-3
    batch_size = 100
    act_noise = 0.01
    start_steps=10000
    update_after=1000
    update_every = 50
    logger_kwargs = dict()
    ac_kwargs = dict()
    num_test_episodes=10
    Params={'name_of_env':name_of_env,'hid':hid,'l':l,'gamma':gamma,'seed':seed,
            'exp_name':exp_name,'replay_size':replay_size,'max_ep_len':max_ep_len,
            'polyak':polyak,'pi_lr':pi_lr,'q_lr':q_lr,'batch_size':batch_size,'act_noise':act_noise,'start_steps':start_steps,
            'update_after':update_after,'update_every':update_every,'logger_kwargs':logger_kwargs,'ac_kwargs':ac_kwargs,
            'num_test_episodes':num_test_episodes,'epochs':epochs}

    return Params, name_of_env, hid, l, epochs,steps_per_epoch, gamma, seed, exp_name, replay_size, max_ep_len, polyak, pi_lr, q_lr, batch_size, act_noise, start_steps, update_after, update_every, num_test_episodes, save_freq


"""
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """