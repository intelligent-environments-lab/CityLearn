import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False


def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True


class NoNorm:
    def __init__(self):
        pass

    def __mul__(self, x):
        return x

    def __rmul__(self, x):
        return x


class PeriodicNorm:
    def __init__(self, x_max):
        self.x_max = x_max

    def __mul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin + 1) / 2.0, (x_cos + 1) / 2.0])

    def __rmul__(self, x):
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
        return np.array([(x_sin + 1) / 2.0, (x_cos + 1) / 2.0])


class OnehotEncode:
    def __init__(self, classes):
        self.classes = classes

    def __mul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]

    def __rmul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]


class Normalize:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __mul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min) / (self.x_max - self.x_min)

    def __rmul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min) / (self.x_max - self.x_min)


class RemoveFeature:
    def __init__(self):
        pass

    def __mul__(self, x):
        return None

    def __rmul__(self, x):
        return None


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
