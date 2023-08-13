#***** HOW TO DEFINE A CUSTOM REWARD FUNCTION *****
#
# This module provides an example of how a custom
# reward function class can be defined by 
# inheriting the base RewardFunction class. Point 
# to this module in reward_function:type object in 
# schema.json to use this custom reward function
# during simulation.
#
#********************** END ***********************

from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction

class CustomReward(RewardFunction):
    """Calculates custom user-defined multi-agent reward.
        
    Reward is the :py:attr:`net_electricity_consumption_emission`
    for entire district if central agent setup otherwise it is the
    :py:attr:`net_electricity_consumption_emission` each building.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
 
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        net_electricity_consumption_emission = [o['net_electricity_consumption_emission'] for o in observations]

        if self.central_agent:
            reward = [-sum(net_electricity_consumption_emission)]
        else:
            reward = [-v for v in net_electricity_consumption_emission]

        return reward