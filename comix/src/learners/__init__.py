from .cq_learner import CQLearner
from .facmaddpg_learner import FacMADDPGLearner
from .maddpg_learner import MADDPGLearner

REGISTRY = {}

REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmaddpg_learner"] = FacMADDPGLearner
REGISTRY["maddpg_learner"] = MADDPGLearner