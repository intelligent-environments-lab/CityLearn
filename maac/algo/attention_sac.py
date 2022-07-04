import torch
from torch.optim import Adam
from ..utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from ..utils.agents import AttentionAgent
from ..utils.critic import AttentionCritic

MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent task
    """
    def __init__(self):
        """
        Inputs:
        """
        pass

    @property
    def policies(self):
        return

    @property
    def target_policies(self):
        return

    def step(self):
        return

