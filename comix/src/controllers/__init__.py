REGISTRY = {}

from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC