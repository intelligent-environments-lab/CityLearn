'''
## Agent File ##

Implementation of Deep Distributed Distributional Deterministic Policy Gradients (D4PG) network
using Tensorflow.
See https://arxiv.org/pdf/1804.08617.pdf for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) 

Based and modified from code by Mark Sinton at https://github.com/msinto93/D4PG 
'''
# LOAD MODULES
import numpy as np
import os
import sys
import scipy.stats as ss
import random
import operator

import tensorflow as tf
from collections import deque

class train_params():
    
    # Environment parameters
    RANDOM_SEED = 99999999                  # Random seed for reproducability
    NUM_AGENTS = 4                          # Number of distributed agents to run simultaneously
     
    V_MIN = -250000.0
    V_MAX = -10000.0
        
    # Training parameters
    BATCH_SIZE = 2048
    NUM_STEPS_TRAIN = 40000       # Number of steps to train for
    MAX_EP_LENGTH = 8759          # Maximum number of steps per episode
    REPLAY_MEM_SIZE = 20000       # Soft maximum capacity of replay memory
    REPLAY_MEM_REMOVE_STEP = 200    # Check replay memory every REPLAY_MEM_REMOVE_STEP training steps and remove samples over REPLAY_MEM_SIZE capacity
    PRIORITY_ALPHA = 0.6            # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    PRIORITY_BETA_START = 0.4       # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
    PRIORITY_BETA_END = 1.0         # Beta will be linearly annealed from its start value to this value throughout training
    PRIORITY_EPSILON = 0.00001      # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
    NOISE_SCALE = 0.3               # Scaling to apply to Gaussian noise
    NOISE_DECAY = 0.9999            # Decay noise throughout training by scaling by noise_decay**training_step
    DISCOUNT_RATE = 0.99            # Discount rate (gamma) for future rewards
    N_STEP_RETURNS = 5              # Number of future steps to collect experiences for N-step returns
    UPDATE_AGENT_EP = 2            # Agent gets latest parameters from learner every update_agent_ep episodes
        
    # Network parameters
    CRITIC_LEARNING_RATE = 0.0001
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_L2_LAMBDA = 0.0          # Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed
    DENSE1_SIZE = 400               # Size of first hidden layer in networks
    DENSE2_SIZE = 300               # Size of second hidden layer in networks
    FINAL_LAYER_INIT = 0.003        # Initialise networks' final layer weights in range +/-final_layer_init
    NUM_ATOMS = 51                  # Number of atoms in output layer of distributional critic
    TAU = 0.001                     # Parameter for soft target network updates
    USE_BATCH_NORM = False          # Whether or not to use batch normalisation in the networks
        
    # Files/Directories
    SAVE_CKPT_STEP = 8760                  # Save checkpoint every save_ckpt_step training steps
    CKPT_DIR = './ckpts'          # Directory for saving/loading checkpoints
    CKPT_FILE = 'citylearn.ckpt-17520'                        # Checkpoint file to load and resume training from (if None, train from scratch)
    LOG_DIR = './logs/train'      # Directory for saving Tensorboard logs (if None, do not save logs)

class test_params:
   
    # Environment parameters
    RANDOM_SEED = 999999                                    # Random seed for reproducability
    
    # Testing parameters
    MAX_EP_LENGTH = 8759                                   # Maximum number of steps per episode
    
    # Files/directories
    CKPT_DIR = './ckpts'                             # Directory for saving/loading checkpoints
    CKPT_FILE = None                                        # Checkpoint file to load and test (if None, load latest ckpt)
    RESULTS_DIR = './test_results'                          # Directory for saving txt file of results (if None, do not save results)
    LOG_DIR = './logs/test'                          # Directory for saving Tensorboard logs (if None, do not save logs)

class play_params:
   
    # Environment parameters
    RANDOM_SEED = 999999                                    # Random seed for reproducability
    
    # Play parameters
    MAX_EP_LENGTH = 8759                                  # Maximum number of steps per episode
    
    # Files/directories
    CKPT_DIR = './ckpts'                             # Directory for saving/loading checkpoints
    CKPT_FILE = None#'citylearn.ckpt-87600'                         # Checkpoint file to load and run (if None, load latest ckpt)


'''
## Learner ##
# Learner class - this trains the D4PG network on experiences sampled (by priority) from the PER buffer
'''
class Learner:
    def __init__(self, env, sess, PER_memory, run_agent_event, stop_agent_event):
        print("Initialising learner... \n\n")
        
        self.sess = sess
        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event
        
        self.observations_spaces, self.actions_spaces = env.get_state_action_spaces()
        self.STATE_DIMS = self.observations_spaces[0].shape
        self.ACTION_DIMS = self.actions_spaces[0].shape
        self.ACTION_BOUND_LOW  = env.action_space.low
        self.ACTION_BOUND_HIGH = env.action_space.high
        
    def build_network(self):
        
        # Define input placeholders    
        self.state_ph = tf.compat.v1.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + self.STATE_DIMS))
        self.action_ph = tf.compat.v1.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + self.ACTION_DIMS))
        self.target_atoms_ph = tf.compat.v1.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS)) # Atom values of target network with Bellman update applied
        self.target_Z_ph = tf.compat.v1.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS))  # Future Z-distribution - for critic training
        self.action_grads_ph = tf.compat.v1.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + self.ACTION_DIMS)) # Gradient of critic's value output wrt action input - for actor training
        self.weights_ph = tf.compat.v1.placeholder(tf.float32, (train_params.BATCH_SIZE)) # Batch of IS weights to weigh gradient updates based on sample priorities

        # Create value (critic) network + target network
        if train_params.USE_BATCH_NORM:
            self.critic_net = Critic_BN(self.state_ph, self.action_ph, self.STATE_DIMS, self.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_main')
            self.critic_target_net = Critic_BN(self.state_ph, self.action_ph, self.STATE_DIMS, self.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_target')
        else:
            self.critic_net = Critic(self.state_ph, self.action_ph, self.STATE_DIMS, self.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_main')
            self.critic_target_net = Critic(self.state_ph, self.action_ph, self.STATE_DIMS, self.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_target')
        
        # Create policy (actor) network + target network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_main')
            self.actor_target_net = Actor_BN(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, self.ACTION_BOUND_LOW, self.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_target')
        else:
            self.actor_net = Actor(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, self.ACTION_BOUND_LOW, self.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_main')
            self.actor_target_net = Actor(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, self.ACTION_BOUND_LOW, self.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_target')
     
        # Create training step ops
        self.critic_train_step = self.critic_net.train_step(self.target_Z_ph, self.target_atoms_ph, self.weights_ph, train_params.CRITIC_LEARNING_RATE, train_params.CRITIC_L2_LAMBDA)
        self.actor_train_step = self.actor_net.train_step(self.action_grads_ph, train_params.ACTOR_LEARNING_RATE, train_params.BATCH_SIZE)
        
        # Create saver for saving model ckpts (we only save learner network vars)
        model_name = 'citylearn.ckpt'
        self.checkpoint_path = os.path.join(train_params.CKPT_DIR, model_name)        
        if not os.path.exists(train_params.CKPT_DIR):
            os.makedirs(train_params.CKPT_DIR)
        saver_vars = [v for v in tf.compat.v1.global_variables() if 'learner' in v.name]
        self.saver = tf.compat.v1.train.Saver(var_list = saver_vars, max_to_keep=201) 
        
    def build_update_ops(self):     
        network_params = self.actor_net.network_params + self.critic_net.network_params
        target_network_params = self.actor_target_net.network_params + self.critic_target_net.network_params
        
        # Create ops which update target network params with hard copy of main network params
        init_update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            init_update_op.append(to_var.assign(from_var))
        
        # Create ops which update target network params with fraction of (tau) main network params
        update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            update_op.append(to_var.assign((tf.multiply(from_var, train_params.TAU) + tf.multiply(to_var, 1. - train_params.TAU))))        
            
        self.init_update_op = init_update_op
        self.update_op = update_op
    
    def initialise_vars(self):
        # Load ckpt file if given, otherwise initialise variables and hard copy to target networks
        if train_params.CKPT_FILE is not None:
            #Restore all learner variables from ckpt
            ckpt = train_params.CKPT_DIR + '/' + train_params.CKPT_FILE
            ckpt_split = ckpt.split('-')
            step_str = ckpt_split[-1]
            self.start_step = int(step_str)    
            self.saver.restore(self.sess, ckpt)
        else:
            self.start_step = 0
            self.sess.run(tf.compat.v1.global_variables_initializer())   
            # Perform hard copy (tau=1.0) of initial params to target networks
            self.sess.run(self.init_update_op)
            
    def run(self):
        # Sample batches of experiences from replay memory and train learner networks 
            
        # Initialise beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END - train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN
        
        # Can only train when we have at least batch_size num of samples in replay memory
        while len(self.PER_memory) <= train_params.BATCH_SIZE:
            sys.stdout.write('\rPopulating replay memory up to batch_size samples...')   
            sys.stdout.flush()
        
        # Training
        sys.stdout.write('\n\nTraining...\n')   
        sys.stdout.flush()
    
        for train_step in range(self.start_step+1, train_params.NUM_STEPS_TRAIN+1):  
            # Get minibatch
            minibatch = self.PER_memory.sample(train_params.BATCH_SIZE, priority_beta) 
            
            states_batch = minibatch[0]
            actions_batch = minibatch[1]
            rewards_batch = minibatch[2]
            next_states_batch = minibatch[3]
            terminals_batch = minibatch[4]
            gammas_batch = minibatch[5]
            weights_batch = minibatch[6]
            idx_batch = minibatch[7]            
    
            # Critic training step    
            # Predict actions for next states by passing next states through policy target network
            future_action = self.sess.run(self.actor_target_net.output, {self.state_ph:next_states_batch})  
            # Predict future Z distribution by passing next states and actions through value target network, also get target network's Z-atom values
            target_Z_dist, target_Z_atoms = self.sess.run([self.critic_target_net.output_probs, self.critic_target_net.z_atoms], {self.state_ph:next_states_batch, self.action_ph:future_action})
            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0)
            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0
            # Apply Bellman update to each atom
            target_Z_atoms = np.expand_dims(rewards_batch, axis=1) + (target_Z_atoms*np.expand_dims(gammas_batch, axis=1))
            # Train critic
            TD_error, _ = self.sess.run([self.critic_net.loss, self.critic_train_step], {self.state_ph:states_batch, self.action_ph:actions_batch, self.target_Z_ph:target_Z_dist, self.target_atoms_ph:target_Z_atoms, self.weights_ph:weights_batch})   
            # Use critic TD errors to update sample priorities
            self.PER_memory.update_priorities(idx_batch, (np.abs(TD_error)+train_params.PRIORITY_EPSILON))
                        
            # Actor training step
            # Get policy network's action outputs for selected states
            actor_actions = self.sess.run(self.actor_net.output, {self.state_ph:states_batch})
            # Compute gradients of critic's value output distribution wrt actions
            action_grads = self.sess.run(self.critic_net.action_grads, {self.state_ph:states_batch, self.action_ph:actor_actions})
            # Train actor
            self.sess.run(self.actor_train_step, {self.state_ph:states_batch, self.action_grads_ph:action_grads[0]})
            
            # Update target networks
            self.sess.run(self.update_op)
            
            # Increment beta value at end of every step   
            priority_beta += beta_increment
                            
            # Periodically check capacity of replay mem and remove samples (by FIFO process) above this capacity
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.PER_memory) > train_params.REPLAY_MEM_SIZE:
                    # Prevent agent from adding new experiences to replay memory while learner removes samples
                    self.run_agent_event.clear()
                    samples_to_remove = len(self.PER_memory) - train_params.REPLAY_MEM_SIZE
                    self.PER_memory.remove(samples_to_remove)
                    # Allow agent to continue adding experiences to replay memory
                    self.run_agent_event.set()
                    
            sys.stdout.write('\rStep {:d}/{:d}'.format(train_step, train_params.NUM_STEPS_TRAIN))
            sys.stdout.flush()  
            
            # Save ckpt periodically
            if train_step % train_params.SAVE_CKPT_STEP == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=train_step)
                sys.stdout.write('\nCheckpoint saved.\n')   
                sys.stdout.flush() 
        
        # Stop the agents
        self.stop_agent_event.set()
        
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0][0]
        
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])

        self.action_tracker.append(a)
        return np.array(a)

'''
## Agent ##
# Agent class - the agent explores the environment, collecting experiences and adding them to the PER buffer. 
Can also be used to test/run a trained network in the environment.
'''
class Agent:
    def __init__(self, sess, env, seed, training, ckpt_dir = None, ckpt_file = None, n_agent=0):
        print("Initialising agent %02d... \n" % n_agent)

        
        if sess == None:
            # Set random seeds for reproducability
            np.random.seed(play_params.RANDOM_SEED)
            tf.set_random_seed(play_params.RANDOM_SEED)
    
            tf.reset_default_graph()

            # Create session
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 

        self.sess = sess        
        self.n_agent = n_agent
        self.env = env
        
        self.observations_spaces, self.actions_spaces = env.get_state_action_spaces()
        self.STATE_DIMS = self.observations_spaces[0].shape
        self.ACTION_DIMS = self.actions_spaces[0].shape
        self.ACTION_BOUND_LOW  = env.action_space.low
        self.ACTION_BOUND_HIGH = env.action_space.high
       
        # Create environment    
        self.env.seed(seed*(n_agent+1))
        
        # BUILD NETWORK
        # Input placeholder    
        self.state_ph = tf.placeholder(tf.float32, ((None,) + self.STATE_DIMS)) 
        
        if training:
            # each agent has their own var_scope
            var_scope = ('actor_agent_%02d'%self.n_agent)
        else:
            # when testing, var_scope comes from main learner policy (actor) network
            var_scope = ('learner_actor_main')
          
        # Create policy (actor) network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, self.ACTION_BOUND_LOW, self.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=False, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params + self.actor_net.bn_params
        else:
            self.actor_net = Actor(self.state_ph, self.STATE_DIMS, self.ACTION_DIMS, self.ACTION_BOUND_LOW, self.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params
        
        # LOAD CHECKPOINT IF TESTING
        if training == False:
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
            
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
            self.ckpt = ckpt
                        
    def build_update_op(self, learner_policy_params):
        # Update agent's policy network params from learner
        update_op = []
        from_vars = learner_policy_params
        to_vars = self.agent_policy_params
                
        for from_var,to_var in zip(from_vars,to_vars):
            update_op.append(to_var.assign(from_var))
        
        self.update_op = update_op
                        
    def build_summaries(self, logdir):
        # Create summary writer to write summaries to disk
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        
        # Create summary op to save episode reward to Tensorboard log
        self.ep_reward_var = tf.Variable(0.0, trainable=False, name=('ep_reward_agent_%02d'%self.n_agent))
        tf.summary.scalar("Episode_Reward", self.ep_reward_var)
        self.summary_op = tf.summary.merge_all()
        
        # Initialise reward var - this will not be initialised with the other network variables as these are copied over from the learner
        self.init_reward_var = tf.variables_initializer([self.ep_reward_var])
            
    def run(self, PER_memory, gaussian_noise, run_agent_event, stop_agent_event):
        # Continuously run agent in environment to collect experiences and add to replay memory
                
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()
        
        # Perform initial copy of params from learner to agent
        self.sess.run(self.update_op)
        
        # Initialise var for logging episode reward
        if train_params.LOG_DIR is not None:
            self.sess.run(self.init_reward_var)
        
        # Initially set threading event to allow agent to run until told otherwise
        run_agent_event.set()
        
        num_eps = 0
        
        while not stop_agent_event.is_set():
            num_eps += 1
            # Reset environment and experience buffer
            state = self.env.reset()
            self.exp_buffer.clear()
            
            num_steps = 0
            episode_reward = 0
            ep_done = False
            
            while not ep_done:
                num_steps += 1
                #print(num_steps)
                ## Take action and store experience
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                action += (gaussian_noise() * train_params.NOISE_DECAY**num_eps)
                next_state, reward, terminal, _ = self.env.step(action)
                
                episode_reward += reward 
                
                self.exp_buffer.append((state, action, reward))
                
                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE
                    
                    # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                    run_agent_event.wait()   
                    PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                
                state = next_state
                
                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Log total episode reward
                    if train_params.LOG_DIR is not None:
                        summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: episode_reward})
                        self.summary_writer.add_summary(summary_str, num_eps)
                    # Compute Bellman rewards and add experiences to replay memory for the last N-1 experiences still remaining in the experience buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE
                        
                        # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                        run_agent_event.wait()     
                        PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                    
                    # Start next episode
                    ep_done = True
                
            # Update agent networks with learner params every 'update_agent_ep' episodes
            if num_eps % train_params.UPDATE_AGENT_EP == 0:
                self.sess.run(self.update_op)
        
        self.env.close()
    
    def load_ckpt(self, ckpt_dir, ckpt_file):
        # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
        loader = tf.train.Saver()    
        if ckpt_file is not None:
            ckpt = ckpt_dir + '/' + ckpt_file  
        else:
            ckpt = tf.train.latest_checkpoint(ckpt_dir)
        
        loader.restore(self.sess, ckpt)
        sys.stdout.write('%s restored.\n\n' % ckpt)
        sys.stdout.flush() 
             
        ckpt_split = ckpt.split('-')
        self.train_ep = ckpt_split[-1]
            
        return ckpt
    
    def test(self):   
        # Test a saved ckpt of actor network and save results to file (optional)
        
        # Create Tensorboard summaries to save episode rewards
        if test_params.LOG_DIR is not None:
            self.build_summaries(test_params.LOG_DIR)

        state = self.env.reset()
        ep_reward = 0
        step = 0
        ep_done = False
            
        while not ep_done:
            action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
            state, reward, terminal, _ = self.env.step(action)

            ep_reward += reward
            step += 1
                 
            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == test_params.MAX_EP_LENGTH:  
                rewards = ep_reward
                ep_done = True   
                
        sys.stdout.write('\x1b[2K\rTesting complete \t Episode reward = {:.2f} \n\n'.format(rewards))
        sys.stdout.write('\x1b[2K\rRamping = {:f} \n'.format(self.env.cost()["ramping"]))
        sys.stdout.write('\x1b[2K\r1-load_factor = {:f} \n'.format(self.env.cost()["1-load_factor"]))
        sys.stdout.write('\x1b[2K\raverage_daily_peak = {:f} \n'.format(self.env.cost()["average_daily_peak"]))
        sys.stdout.write('\x1b[2K\rpeak_demand = {:f} \n'.format(self.env.cost()["peak_demand"]))
        sys.stdout.write('\x1b[2K\rnet_electricity_consumption = {:f} \n'.format(self.env.cost()["net_electricity_consumption"]))
        sys.stdout.write('\x1b[2K\rtotal = {:f} \n'.format(self.env.cost()["total"]))
        sys.stdout.flush()  
        
        # Log average episode reward for Tensorboard visualisation
        if test_params.LOG_DIR is not None:
            summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: rewards})
            self.summary_writer.add_summary(summary_str, self.train_ep)
         
        # Write results to file        
        if test_params.RESULTS_DIR is not None:
            if not os.path.exists(test_params.RESULTS_DIR):
                os.makedirs(test_params.RESULTS_DIR)
            output_file = open(test_params.RESULTS_DIR + '/citylearn' + '.txt' , 'a')
            output_file.write('Training Episode {}: \t Episode reward = {:.2f} \n\n'.format(self.train_ep, rewards))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()      
        
        self.env.close()
'''
## Prioritised Experience Replay (PER) Memory ##
# Adapted from: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Creates prioritised replay memory buffer to add experiences to and sample batches of experiences from
'''
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)

        self._storage.append(data)
        
        self._next_idx += 1
        
    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(gammas)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size*2:     # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"
        
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        
    def remove(self, num_samples):
        super().remove(num_samples)  
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

'''
## Gaussian Noise ##
# Creates Gaussian noise process for adding exploration noise to the action space during training
Adapted from https://github.com/msinto93/D4PG/blob/master/utils/gaussian_noise.py
@author: Mark Sinton (msinto93@gmail.com).
'''
class GaussianNoiseGenerator:
    def __init__(self, action_dims, action_bound_low, action_bound_high, noise_scale):
        assert np.array_equal(np.abs(action_bound_low), action_bound_high)
        
        self.action_dims = action_dims
        self.action_bounds = action_bound_high
        self.scale = noise_scale

    def __call__(self):
        noise = np.random.normal(size=self.action_dims) * self.action_bounds * self.scale
        
        return noise

'''
## Network ##
# Defines the D4PG Value (critic) and Policy (Actor) networks - with and without batch norm
Adapted from https://github.com/msinto93/D4PG/blob/master/utils/network.py
@author: Mark Sinton (msinto93@gmail.com) 
'''
class Critic:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Z distribution should be predicted
         
        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.scope = scope    
         
        with tf.variable_scope(self.scope):           
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
                         
            self.dense1 = relu(self.dense1_mul, scope='dense1')
             
            #Merge first dense layer with action input to get second dense layer            
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
             
            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
                           
            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')
                          
            self.output_logits = dense(self.dense2, num_atoms, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output_logits')  
            
            self.output_probs = softmax(self.output_logits, scope='output_probs')
                         
                          
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [] # No batch norm params
            
            
            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)
            
            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs) # the Q value is the mean of the categorical output Z-distribution
          
            self.action_grads = tf.gradients(self.output_probs, self.action, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
            

    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)               
                
                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)  
                
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                                                
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss
                 
                train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)
                  
                return train_step
        

class Actor:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
         
        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.scope = scope
        
        with tf.variable_scope(self.scope):
                    
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
                         
            self.dense1 = relu(self.dense1_mul, scope='dense1')
             
            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
                         
            self.dense2 = relu(self.dense2_mul, scope='dense2')
             
            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
             
            self.output_tanh = tanh(self.output_mul, scope='output')
             
            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
             
            
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [] # No batch norm params
        
        
    def train_step(self, action_grads, learn_rate, batch_size):
        # action_grads - gradient of value output wrt action from critic network
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                 
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
                 
                train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
                 
                return train_step
            
            
class Critic_BN:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, is_training=False, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Z distribution should be predicted
        
        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.is_training = is_training
        self.scope = scope    

        
        with tf.variable_scope(self.scope):
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')
           
            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
            
            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')
            
            self.dense1 = relu(self.dense1_bn, scope='dense1')
            
            #Merge first dense layer with action input to get second dense layer            
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
            
            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
            
            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')
            
            self.output_logits = dense(self.dense2, num_atoms, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output_logits')  
            
            self.output_probs = softmax(self.output_logits, scope='output_probs')
                         
                          
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
            
            
            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)
            
            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs) # the Q value is the mean of the categorical output Z-distribution
          
            self.action_grads = tf.gradients(self.output_probs, self.action, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
            
    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)               
                
                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)  
                
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                                                
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)
                 
                return train_step
        

class Actor_BN:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, is_training=False, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
        
        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.is_training = is_training
        self.scope = scope
        
        with tf.variable_scope(self.scope):
        
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')
           
            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
            
            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')
            
            self.dense1 = relu(self.dense1_bn, scope='dense1')
            
            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
            
            self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, scope='dense2')
            
            self.dense2 = relu(self.dense2_bn, scope='dense2')
            
            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
            
            self.output_tanh = tanh(self.output_mul, scope='output')
            
            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
            
           
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
        
    def train_step(self, action_grads, learn_rate, batch_size):
        # action_grads - gradient of value output wrt action from critic network
        
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
                
                return train_step

'''
## Ops ##
# Common ops for the networks
Adapted from https://github.com/msinto93/D4PG/blob/master/utils/ops.py
@author: Mark Sinton (msinto93@gmail.com) 
'''
def conv2d(inputs, kernel_size, filters, stride, activation=None, use_bias=True, weight_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer(), scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=weight_init)
        else:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=weight_init,
                                    bias_initializer=bias_init)
            
def batchnorm(inputs, is_training, momentum=0.9, scope='batch_norm'):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, momentum=momentum, training=is_training, fused=True)

def dense(inputs, output_size, activation=None, weight_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer(), scope='dense'):
    with tf.variable_scope(scope):
        return tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=weight_init, bias_initializer=bias_init)

def flatten(inputs, scope='flatten'):
    with tf.variable_scope(scope):
        return tf.layers.flatten(inputs)
    
def relu(inputs, scope='relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(inputs)
    
def tanh(inputs, scope='tanh'):
    with tf.variable_scope(scope):
        return tf.nn.tanh(inputs)
    
def softmax(inputs, scope='softmax'):
    with tf.variable_scope(scope):
        return tf.nn.softmax(inputs)

'''
## l2_projection ##
# Taken from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
# Projects the target distribution onto the support of the original network [Vmin, Vmax]
'''    
def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).
    
    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp
    
    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1
    
    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1
    
    delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp
    
    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)

'''
## Segment Tree ##
# Adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
# Segment tree data structures used to store the priorities of the samples in the PER for efficient priority-based sampling. 
'''
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self.neutral_element = neutral_element
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]
    
    def remove_items(self, num_items):
        """Removes num_items leaf nodes from the tree and appends num_items leaf nodes of neutral_element value (0 or inf) to end of tree,
           effectively left shifting remaining leaf nodes by num_items"""
        del self._value[self._capacity:(self._capacity + num_items)]
        neutral_elements = [self.neutral_element for _ in range(num_items)]
        self._value += neutral_elements
        for idx in range(self._capacity-1, 0, -1):
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
                )
                            

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)