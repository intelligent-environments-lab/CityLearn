import math
from typing import List
import numpy as np
from citylearn.agents.base import Agent

class TabularQLearning(Agent):
    def __init__(
        self, *args, epsilon: float = None, minimum_epsilon: float = None, epsilon_decay: float = None, 
        learning_rate: float = None, discount_factor: float = None, q_init_value: float = None, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = 1.0 if epsilon is None else epsilon
        self.epsilon_init = self.epsilon
        self.minimum_epsilon = 0.01 if minimum_epsilon is None else minimum_epsilon
        self.epsilon_decay = 0.0001 if epsilon_decay is None else epsilon_decay
        self.learning_rate = 0.05 if learning_rate is None else learning_rate
        self.discount_factor = 0.90 if discount_factor is None else discount_factor
        self.q_init_value = np.nan if q_init_value is None else q_init_value
        self.q, self.q_exploration, self.q_exploitation = self.__initialize_q()
        self.__explored = False

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:    
        deterministic = False if deterministic is None else deterministic
        actions = None
        seed = self.random_seed if self.random_seed is None else self.random_seed + self.time_step
        np.random.seed(seed)
        

        if deterministic or np.random.random() > self.epsilon:
            # Use q-function to decide action
            actions = self.exploit(observations)
            self.__explored = False
        
        else:
            # Explore random action
            actions = [[s.sample()] for s in self.env.action_space]
            self.__explored = True

        # exponential decay
        episode = int(self.time_step/self.env.time_steps)
        self.epsilon = max(self.minimum_epsilon, self.epsilon_init*np.exp(-self.epsilon_decay*episode))

        self.actions = actions
        self.next_time_step()
        return actions

    def exploit(self, observations: List[List[float]]) -> List[List[float]]:
        actions = []

        for i, o in enumerate(observations):
            o = o[0]

            try:
                a = np.nanargmax(self.q[i][o])
            
            except ValueError:
                # when all values for observation are still NaN
                a = self.env.action_space[i].sample()

            actions.append([a])
        
        return actions

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], done: bool):
        # Compute temporal difference target and error to udpate q-function
        
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o, n, a = o[0], n[0], a[0]
            current_q = self.q[i][o, a]
            current_q = 0.0 if math.isnan(current_q) else current_q

            try:
                next_max_q = np.nanargmax(self.q[i][n])
            
            except ValueError:
                # when all values for observation are still NaN
                next_max_q = 0.0
            
            # update q
            new_q = current_q + self.learning_rate*(r + self.discount_factor*next_max_q - current_q)
            self.q[i][o, a] = new_q

            # update exploration-exploitation count
            if self.__explored:
                self.q_exploration[i][o, a] += 1
            else:
                self.q_exploitation[i][o, a] += 1

    def __initialize_q(self) -> np.ndarray:
        q = [None for _ in self.env.observation_space]
        q_exploration = [None for _ in self.env.observation_space]
        q_exploitation = [None for _ in self.env.observation_space]

        for i, (od, ad) in enumerate(zip(self.env.observation_space, self.env.action_space)):
            shape = (od.n, ad.n)
            q[i] = np.ones(shape=shape)*self.q_init_value
            q_exploration[i] = np.zeros(shape=shape)
            q_exploitation[i] = np.zeros(shape=shape)
        
        return q, q_exploration, q_exploitation