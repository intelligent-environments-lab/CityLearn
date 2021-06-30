REGISTRY = {}

from .comix_agent import CEMAgent, NAFAgent
from .mlp_agent import MLPAgent

REGISTRY["naf"] = NAFAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["mlp"] = MLPAgent