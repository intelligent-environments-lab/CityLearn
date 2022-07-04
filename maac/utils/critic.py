import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets itw own
    observation and action, and can also attend over the other agent's encoded
    observations and actions.
    """
    def __init__(self):
        """
        Inputs:
        """
        super(AttentionCritic,self).__init__()