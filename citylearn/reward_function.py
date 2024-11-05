from typing import Any, List, Mapping, Tuple, Union
import numpy as np
from citylearn.building import Building
from citylearn.data import ZERO_DIVISION_PLACEHOLDER

class RewardFunction:
    r"""Base and default reward function class.

    The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], exponent: float = None, **kwargs):
        self.env_metadata = env_metadata
        self.exponent = exponent

    @property
    def env_metadata(self) -> Mapping[str, Any]:
        """General static information about the environment."""

        return self.__env_metadata
    
    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all buildings."""

        return self.env_metadata['central_agent']
    
    @property
    def exponent(self) -> float:
        return self.__exponent
    
    @env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        self.__env_metadata = env_metadata

    @exponent.setter
    def exponent(self, exponent: float):
        self.__exponent = 1.0 if exponent is None else exponent

    def reset(self):
        """Use to reset variables at the start of an episode."""

        pass

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

        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        reward_list = [-(max(o, 0)**self.exponent) for o in net_electricity_consumption]

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward

class MARL(RewardFunction):
    """MARL reward function class.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        district_electricity_consumption = sum(net_electricity_consumption)
        building_electricity_consumption = np.array(net_electricity_consumption, dtype=float)*-1
        reward_list = np.sign(building_electricity_consumption)*0.01*building_electricity_consumption**2*np.nanmax([0, district_electricity_consumption])

        if self.central_agent:
            reward = [reward_list.sum()]
        else:
            reward = reward_list.tolist()
        
        return reward

class IndependentSACReward(RewardFunction):
    """Recommended for use with the `SAC` controllers.
    
    Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        reward_list = [min(v*-1**3, 0) for v in net_electricity_consumption]

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward
    
class SolarPenaltyReward(RewardFunction):
    """The reward is designed to minimize electricity consumption and maximize solar generation to charge energy storage systems.

    The reward is calculated for each building, i and summed to provide the agent with a reward that is representative of all the
    building or buildings (in centralized case)it controls. It encourages net-zero energy use by penalizing grid load satisfaction 
    when there is energy in the energy storage systems as well as penalizing net export when the energy storage systems are not
    fully charged through the penalty term. There is neither penalty nor reward when the energy storage systems are fully charged
    during net export to the grid. Whereas, when the energy storage systems are charged to capacity and there is net import from the 
    grid the penalty is maximized.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):
            e = o['net_electricity_consumption']
            cc = m['cooling_storage']['capacity']
            hc = m['heating_storage']['capacity']
            dc = m['dhw_storage']['capacity']
            ec = m['electrical_storage']['capacity']
            cs = o.get('cooling_storage_soc', 0.0)
            hs = o.get('heating_storage_soc', 0.0)
            ds = o.get('dhw_storage_soc', 0.0)
            es = o.get('electrical_storage_soc', 0.0)
            reward = 0.0
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list
        
        return reward
    
class ComfortReward(RewardFunction):
    """Reward for occupant thermal comfort satisfaction.

    The reward is calculated as the negative difference between the setpoint and indoor dry-bulb temperature raised to some exponent
    if outside the comfort band. If within the comfort band, the reward is the negative difference when in cooling mode and temperature
    is below the setpoint or when in heating mode and temperature is above the setpoint. The reward is 0 if within the comfort band
    and above the setpoint in cooling mode or below the setpoint and in heating mode.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default: 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env_metadata)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent

    @property
    def band(self) -> float:
        return self.__band
    
    @property
    def lower_exponent(self) -> float:
        return self.__lower_exponent
    
    @property
    def higher_exponent(self) -> float:
        return self.__higher_exponent
    
    @band.setter
    def band(self, band: float):
        self.__band = band

    @lower_exponent.setter
    def lower_exponent(self, lower_exponent: float):
        self.__lower_exponent = 2.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 2.0 if higher_exponent is None else higher_exponent

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            hvac_mode = o['hvac_mode']
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']

            if hvac_mode in [1, 2]:
                set_point = o['indoor_dry_bulb_temperature_cooling_set_point'] if hvac_mode == 1 else o['indoor_dry_bulb_temperature_heating_set_point']
                band =  self.band if self.band is not None else o['comfort_band']
                lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - band
                upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + band
                delta = abs(indoor_dry_bulb_temperature - set_point)
                
                if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                    exponent = self.lower_exponent if hvac_mode == 2 else self.higher_exponent
                    reward = -(delta**exponent)
                
                elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                    reward = 0.0 if heating else -delta

                elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                    reward = -delta if heating else 0.0

                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(delta**exponent)

            else:
                cooling_set_point = o['indoor_dry_bulb_temperature_cooling_set_point']
                heating_set_point = o['indoor_dry_bulb_temperature_heating_set_point']
                band =  self.band if self.band is not None else o['comfort_band']
                lower_bound_comfortable_indoor_dry_bulb_temperature = heating_set_point - band
                upper_bound_comfortable_indoor_dry_bulb_temperature = cooling_set_point + band
                cooling_delta = indoor_dry_bulb_temperature - cooling_set_point
                heating_delta = indoor_dry_bulb_temperature - heating_set_point

                if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                    exponent = self.higher_exponent if not heating else self.lower_exponent
                    reward = -(abs(heating_delta)**exponent)

                elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < heating_set_point:
                    reward = -(abs(heating_delta))

                elif heating_set_point <= indoor_dry_bulb_temperature <= cooling_set_point:
                    reward = 0.0

                elif cooling_set_point < indoor_dry_bulb_temperature < upper_bound_comfortable_indoor_dry_bulb_temperature:
                    reward = -(abs(cooling_delta))

                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(abs(cooling_delta)**exponent)

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
    
class SolarPenaltyAndComfortReward(RewardFunction):
    """Addition of :py:class:`citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward`.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 3.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    coefficients: Tuple, default = (1.0, 1.0)
        Coefficents for `citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward` values respectively.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None, coefficients: Tuple = None):
        self.__functions: List[RewardFunction] = [
            SolarPenaltyReward(env_metadata),
            ComfortReward(env_metadata, band=band, lower_exponent=lower_exponent, higher_exponent=higher_exponent)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        RewardFunction.env_metadata.fset(self, env_metadata)

        for f in self.__functions:
            f.env_metadata = self.env_metadata
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
        reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()

        return reward


class V2GPenaltyReward(MARL):
    """Rewards with considerations for electric vehicle charging behaviours in a V2G setting.
    Note that this function rewards/penalizes only the electric vehicle part. For a comprehensive reward strategy
    please use one of the super classes or rewrite your own

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any],
                 peak_percentage_threshold : float = None, ramping_percentage_threshold : float = None, peak_penalty_weight : int = None, ramping_penalty_weight : int = None,
                 energy_transfer_bonus : int = None, window_size : int = None, penalty_no_car_charging : int = None, penalty_battery_limits : int = None, penalty_soc_under_5_10 : int = None,
                 reward_close_soc : int = None, reward_self_ev_consumption : int = None, community_weight : float = None, reward_extra_self_production : int = None):
        super().__init__(env_metadata)

        # Setting the parameters
        self.peak_percentage_threshold = peak_percentage_threshold
        self.ramping_percentage_threshold = ramping_percentage_threshold
        self.peak_penalty_weight = peak_penalty_weight
        self.ramping_penalty_weight = ramping_penalty_weight
        self.energy_transfer_bonus = energy_transfer_bonus
        self.window_size = window_size
        self.penalty_no_car_charging = penalty_no_car_charging
        self.penalty_battery_limits = penalty_battery_limits
        self.penalty_soc_under_5_10 = penalty_soc_under_5_10
        self.reward_close_soc = reward_close_soc
        self.community_weight = community_weight
        self.reward_extra_self_production = reward_extra_self_production
        self.reward_self_ev_consumption = reward_self_ev_consumption

    @property
    def peak_percentage_threshold(self) -> float:
        """Return the peak_percentage_threshold"""

        return self.__peak_percentage_threshold

    @peak_percentage_threshold.setter
    def peak_percentage_threshold(self, peak_percentage_threshold: float):
        if peak_percentage_threshold is None:
            self.__peak_percentage_threshold = 0.10
        else:
            self.__peak_percentage_threshold = peak_percentage_threshold

    @property
    def ramping_percentage_threshold(self) -> float:
        """Return the ramping_percentage_threshold"""

        return self.__ramping_percentage_threshold

    @ramping_percentage_threshold.setter
    def ramping_percentage_threshold(self, ramping_percentage_threshold: float):
        if ramping_percentage_threshold is None:
            self.__ramping_percentage_threshold = 0.10
        else:
            self.__ramping_percentage_threshold = ramping_percentage_threshold

    @property
    def peak_penalty_weight(self) -> int:
        """Return the peak_penalty_weight"""

        return self.__peak_penalty_weight

    @peak_penalty_weight.setter
    def peak_penalty_weight(self, peak_penalty_weight: int):
        if peak_penalty_weight is None:
            self.__peak_penalty_weight = 20
        else:
            self.__peak_penalty_weight = peak_penalty_weight

    @property
    def ramping_penalty_weight(self) -> int:
        """Return the ramping_penalty_weight"""

        return self.__ramping_penalty_weight

    @ramping_penalty_weight.setter
    def ramping_penalty_weight(self, ramping_penalty_weight: int):
        if ramping_penalty_weight is None:
            self.__ramping_penalty_weight = 15
        else:
            self.__ramping_penalty_weight = ramping_penalty_weight

    @property
    def energy_transfer_bonus(self) -> int:
        """Return the energy_transfer_bonus"""

        return self.__energy_transfer_bonus

    @energy_transfer_bonus.setter
    def energy_transfer_bonus(self, energy_transfer_bonus: int):
        if energy_transfer_bonus is None:
            self.__energy_transfer_bonus = 10
        else:
            self.__energy_transfer_bonus = energy_transfer_bonus

    @property
    def window_size(self) -> int:
        """Return the window_size"""

        return self.__window_size

    @window_size.setter
    def window_size(self, window_size: int):
        if window_size is None:
            self.__window_size = 6
        else:
            self.__window_size = window_size

    @property
    def penalty_no_car_charging(self) -> int:
        """Return the penalty_no_car_charging"""

        return self.__penalty_no_car_charging

    @penalty_no_car_charging.setter
    def penalty_no_car_charging(self, penalty_no_car_charging: int):
        if penalty_no_car_charging is None:
            self.__penalty_no_car_charging = -5
        else:
            self.__penalty_no_car_charging = penalty_no_car_charging

    @property
    def penalty_battery_limits(self) -> int:
        """Return the penalty_battery_limits"""

        return self.__penalty_battery_limits

    @penalty_battery_limits.setter
    def penalty_battery_limits(self, penalty_battery_limits: int):
        if penalty_battery_limits is None:
            self.__penalty_battery_limits = -2
        else:
            self.__penalty_battery_limits = penalty_battery_limits

    @property
    def penalty_soc_under_5_10(self) -> int:
        """Return the penalty_soc_under_5_10"""

        return self.__penalty_soc_under_5_10

    @penalty_soc_under_5_10.setter
    def penalty_soc_under_5_10(self, penalty_soc_under_5_10: int):
        if penalty_soc_under_5_10 is None:
            self.__penalty_soc_under_5_10 = -5
        else:
            self.__penalty_soc_under_5_10 = penalty_soc_under_5_10

    @property
    def reward_close_soc(self) -> int:
        """Return the reward_close_soc"""

        return self.__penalty_soc_under_5_10

    @reward_close_soc.setter
    def reward_close_soc(self, reward_close_soc: int):
        if reward_close_soc is None:
            self._reward_close_soc = 10
        else:
            self.__reward_close_soc = reward_close_soc      

    @property
    def reward_self_ev_consumption(self) -> int:
        """Return the reward_self_ev_consumption"""

        return self.__reward_self_ev_consumption

    @reward_self_ev_consumption.setter
    def reward_self_ev_consumption(self, reward_self_ev_consumption: int):
        if reward_self_ev_consumption is None:
            self._reward_self_ev_consumption = 5
        else:
            self.__reward_self_ev_consumption = reward_self_ev_consumption      

    @property
    def community_weight(self) -> float:
        """Return the community_weight"""

        return self.__community_weight

    @community_weight.setter
    def community_weight(self, community_weight: float):
        if community_weight is None:
            self._community_weight = 0.2
        else:
            self.__community_weight = community_weight   

    @property
    def reward_extra_self_production(self) -> int:
        """Return the reward_extra_self_production"""

        return self.__reward_extra_self_production

    @reward_extra_self_production.setter
    def reward_extra_self_production(self, reward_extra_self_production: int):
        if reward_extra_self_production is None:
            self._reward_extra_self_production = 5
        else:
            self.__reward_extra_self_production = reward_extra_self_production    


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:

        ##net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        current_reward = super.calculate(observations)
        reward_list = []

        for b in self.env.buildings:
            # Building reward calculation
            reward = self.calculate_ev_penalty(b, current_reward)
            reward_list.append(reward)

        # Central agent reward aggregation
        if self.central_agent:
            reward = [reward_list.sum()]
        else:
            reward = reward_list.tolist()

        return reward

    def calculate_ev_penalty(self, b : Building, current_reward : RewardFunction) -> float:
        """Calculate penalties based on EV specific logic."""
        penalty = 0
        penalty_multiplier = abs(current_reward)  # Multiplier for the penalty

        if b.chargers:
            for c in b.chargers:
                last_connected_car = c.past_connected_evs[-2]
                last_charged_value = c.past_charging_action_values[-2]

                # 1. Penalty for charging when no car is present
                if last_connected_car is None and last_charged_value > 0.1 or last_charged_value < 0.1:
                    penalty += self.PENALTY_NO_CAR_CHARGING * penalty_multiplier

                # 3. Penalty for exceeding the battery's limits
                if last_connected_car is not None:
                   if last_connected_car.battery.soc[-2] + last_charged_value > last_connected_car.battery.capacity:
                       penalty += self.PENALTY_BATTERY_LIMITS * penalty_multiplier
                   if last_connected_car.battery.soc[-2] + last_charged_value < last_connected_car.min_battery_soc:
                       penalty += self.PENALTY_BATTERY_LIMITS * penalty_multiplier


                # 4. Penalties (or Reward) for SoC differences
                if last_connected_car is not None:
                    required_soc = last_connected_car.electric_vehicle_simulation.required_soc_departure[-1]
                    actual_soc = last_connected_car.battery.soc[-1]

                    hours_until_departure = last_connected_car.electric_vehicle_simulation.estimated_departure_time[-1]
                    max_possible_charge = c.max_charging_power * hours_until_departure
                    max_possible_discharge = c.max_discharging_power * hours_until_departure

                    soc_diff = ((actual_soc * 100) / last_connected_car.battery.capacity) - required_soc

                    # If the car needs more charge than it currently has and it's impossible to achieve the required SoC
                    if soc_diff > 0 and soc_diff > max_possible_charge:
                        penalty += self.PENALTY_SOC_UNDER_5_10 ** 2 * penalty_multiplier

                    # Adjusted penalties/rewards based on SoC difference at departure
                    if hours_until_departure == 0:
                        if -25 < soc_diff <= -10:
                            penalty += 2 * self.PENALTY_SOC_UNDER_5_10 * penalty_multiplier
                        elif soc_diff <= -25:
                            penalty += self.PENALTY_SOC_UNDER_5_10 ** 3 * penalty_multiplier
                        elif -10 < soc_diff <= 10:
                            penalty += self.REWARD_CLOSE_SOC * penalty_multiplier  # Reward for leaving with SOC close to the requested value

                    if (soc_diff > 0 and soc_diff <= max_possible_charge) or (
                            soc_diff < 0 and abs(soc_diff) <= max_possible_discharge):
                        reward_multiplier = 1 / (
                                hours_until_departure + 0.1)  # Adding 0.1 to prevent division by zero
                        penalty += self.REWARD_CLOSE_SOC * penalty_multiplier * reward_multiplier

                net_energy_before = b.net_electricity_consumption[b.time_step-1]
                # 5. Reward for charging the car during times of extra self-production
                if last_connected_car is not None and last_charged_value > 0 and net_energy_before < 0:
                    penalty += self.REWARD_EXTRA_SELF_PRODUCTION * penalty_multiplier
                elif last_connected_car is not None and last_charged_value < 0 and net_energy_before < 0:
                    penalty += self.REWARD_EXTRA_SELF_PRODUCTION*-0.5 * penalty_multiplier

                # 6. Reward for discharging the car to support building consumption and avoid importing energy
                if last_connected_car is not None and last_charged_value < 0 and net_energy_before > 0:
                    penalty += self.REWARD_SELF_EV_CONSUMPTION * penalty_multiplier
                elif last_connected_car is not None and last_charged_value > 0 and net_energy_before > 0:
                    penalty += self.REWARD_SELF_EV_CONSUMPTION * -0.5 * penalty_multiplier

        return penalty