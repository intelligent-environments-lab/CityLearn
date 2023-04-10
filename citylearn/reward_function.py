from typing import List
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.energy_model import ZERO_DIVISION_CAPACITY

class RewardFunction:
    def __init__(self, env: CityLearnEnv, **kwargs):
        r"""Initialize `Reward`.

        Parameters
        ----------
        env: citylearn.citylearn.CityLearnEnv
            Simulation environment.
        **kwargs : dict
            Other keyword arguments for custom reward calculation.
        """

        self.env = env
        self.kwargs = kwargs

    @property
    def env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__env

    @env.setter
    def env(self, env: CityLearnEnv):
        self.__env = env

    def calculate(self) -> List[float]:
        r"""Calculates default reward.

        The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{min}(-e_0, 0), \dots, \textrm{min}(-e_n, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        if self.env.central_agent:
            reward = [min(self.env.net_electricity_consumption[self.env.time_step]*-1, 0)]
        else:
            reward = [min(b.net_electricity_consumption[b.time_step]*-1, 0) for b in self.env.buildings]

        return reward

class MARL(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Calculates MARL reward.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.

        Notes
        -----
        Reward value is calculated as :math:`\textrm{sign}(-e) \times 0.01(e^2) \times \textrm{max}(0, E)`
        where :math:`e` is the building `electricity_consumption` and :math:`E` is the district `electricity_consumption`.
        """

        district_electricity_consumption = self.env.net_electricity_consumption[self.env.time_step]
        building_electricity_consumption = np.array([b.net_electricity_consumption[b.time_step]*-1 for b in self.env.buildings])
        reward = np.sign(building_electricity_consumption)*0.01*building_electricity_consumption**2*np.nanmax(0, district_electricity_consumption)
        return reward.tolist()

class IndependentSACReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

        Recommended for use with the `SAC` controllers.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{min}(-e_0^3, 0), \dots, \textrm{min}(-e_n^3, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        return [min(b.net_electricity_consumption[b.time_step]*-1**3, 0) for b in self.env.buildings]
    
class SolarPenaltyReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        """The reward is designed to minimize electricity consumption and maximize
        solar generation to charge energy storage systems.

        The reward is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all the building or buildings (in centralized case)
        it controls. It encourages net-zero energy use by penalizing grid load satisfaction 
        when there is energy in the enerygy storage systems as well as penalizing 
        net export when the energy storage systems are not fully charged through the penalty 
        term. There is neither penalty nor reward when the energy storage systems
        are fully charged during net export to the grid. Whereas, when the 
        energy storage systems are charged to capacity and there is net import from the 
        grid the penalty is maximized.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        
        reward_list = []

        for b in self.env.buildings:
            e = b.net_electricity_consumption[-1]
            cc = b.cooling_storage.capacity
            hc = b.heating_storage.capacity
            dc = b.dhw_storage.capacity
            ec = b.electrical_storage.capacity_history[0]
            cs = b.cooling_storage.soc[-1]/cc
            hs = b.heating_storage.soc[-1]/hc
            ds = b.dhw_storage.soc[-1]/dc
            es = b.electrical_storage.soc[-1]/ec
            reward = 0.0
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_CAPACITY else 0.0
            reward_list.append(reward)


        if self.env.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list
        
        return reward