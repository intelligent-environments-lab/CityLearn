import numpy as np
import random
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        push (s, a, r, n_s, d) into the replay buffer
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of size batch_size
        :param batch_size:
        :return:
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
