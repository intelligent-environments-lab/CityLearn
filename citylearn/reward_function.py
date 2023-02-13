from typing import Any, List, Mapping
import numpy as np
from citylearn.citylearn import CityLearnEnv

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

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{min}(-e_0^3, 0), \dots, \textrm{min}(-e_n^3, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        return [min(b.net_electricity_consumption[b.time_step]*-1**3, 0) for b in self.env.buildings]

class BuildingDynamicsReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

        Recommended for use with the `SAC` controllers.

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{min}(-e_0^3, 0), \dots, \textrm{min}(-e_n^3, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        comfort_reward = self.calculate_comfort_reward()
        storage_reward = self.calculate_storage_reward()
        peak_reward = self.calculate_peak_reward()

        if self.env.central_agent:
            reward = [sum(comfort_reward) + sum(storage_reward) + sum(peak_reward)]
        else:
            reward = [sum([c, s, p]) for c, s, p in zip(comfort_reward, storage_reward, peak_reward)]

        return reward

    def calculate_comfort_reward(self) -> List[float]:
        coefficient = 0.12
        comfort_band = 2.0 # C
        rewards = []

        for b in self.env.buildings:
            indoor_dry_bulb_temperature = b.energy_simulation.indoor_dry_bulb_temperature[b.time_step]
            indoor_dry_bulb_temperature_set_point = b.energy_simulation.cooling_dry_bulb_temperature_set_point[b.time_step]
            lower_bound_comfortable_indoor_dry_bulb_temperature = indoor_dry_bulb_temperature_set_point - comfort_band
            upper_bound_comfortable_indoor_dry_bulb_temperature = indoor_dry_bulb_temperature_set_point + comfort_band
            
            if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                reward = -coefficient*(indoor_dry_bulb_temperature_set_point - indoor_dry_bulb_temperature)**3
            
            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < indoor_dry_bulb_temperature_set_point:
                reward = -coefficient*(indoor_dry_bulb_temperature_set_point - indoor_dry_bulb_temperature)

            elif indoor_dry_bulb_temperature_set_point <= indoor_dry_bulb_temperature < upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = 0

            else:
                reward = -coefficient*(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point)**2

            rewards.append(reward)

        return rewards

    def calculate_storage_reward(self) -> List[float]:
        cooling_storage_coefficient = 3
        heating_storage_coefficient = 3
        dhw_storage_coefficient = 2
        electrical_storage_coefficient = 2
        rewards = []

        for b in self.env.buildings:
            reward = 0
            reward += max(0, (b.cooling_storage.soc[b.time_step] - b.cooling_storage.soc[b.time_step - 1])/b.cooling_storage.capacity)*cooling_storage_coefficient
            reward += max(0, (b.heating_storage.soc[b.time_step] - b.heating_storage.soc[b.time_step - 1])/b.heating_storage.capacity)*heating_storage_coefficient
            reward += max(0, (b.dhw_storage.soc[b.time_step] - b.dhw_storage.soc[b.time_step - 1])/b.dhw_storage.capacity)*dhw_storage_coefficient
            reward += max(0, (b.electrical_storage.soc[b.time_step] - b.electrical_storage.soc[b.time_step - 1])/b.electrical_storage.capacity_history[0])*electrical_storage_coefficient
            rewards.append(reward)

        return rewards

    def calculate_peak_reward(self) -> List[float]:
        return [-b.net_electricity_consumption[b.time_step] for b in self.env.buildings]