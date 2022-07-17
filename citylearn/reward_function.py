from typing import Any, List, Mapping
import numpy as np

class RewardFunction:
    def __init__(self, agent_count: int, electricity_consumption: List[float] = None, carbon_emission: List[float] = None, electricity_price: List[float] = None, **kwargs):
        r"""Initialize `Reward`.

        Parameters
        ----------
        agent_count: int
            Number of agents.
        electricity_consumption: List[float]
            Buildings' electricity consumption in [kWh].
        carbon_emission: List[float], optional
            Buildings' carbon emissions in [kg_co2].
        electricity_price: List[float], optional
            Buildings' electricty prices in [$].
        **kwargs : dict
            Other keyword arguments for custom reward calculation.
        """

        self.agent_count = agent_count
        self.electricty_consumption = electricity_consumption
        self.carbon_emission = carbon_emission
        self.electricity_price = electricity_price
        self.kwargs = kwargs

    @property
    def agent_count(self) -> int:
        """Number of agents."""

        return self.__agent_count

    @property
    def electricity_consumption(self) -> List[float]:
        """Buildings' electricity consumption in [kWh]."""

        return self.__electricity_consumption

    @property
    def carbon_emission(self) -> List[float]:
        """Buildings' carbon emissions in [kg_co2]."""

        return self.__carbon_emission

    @property
    def electricity_price(self) -> List[float]:
        """Buildings' electricty prices in [$]."""

        return self.__electricity_price

    @property
    def kwargs(self) ->Mapping[Any,Any]:
        return self.__kwargs

    @agent_count.setter
    def agent_count(self, agent_count: int):
        self.__agent_count = agent_count

    @electricity_consumption.setter
    def electricity_consumption(self, electricity_consumption: List[float]):
        self.__electricity_consumption = [np.nan]*self.agent_count if electricity_consumption is None else electricity_consumption

    @carbon_emission.setter
    def carbon_emission(self, carbon_emission: List[float]):
        self.__carbon_emission = [np.nan]*self.agent_count if carbon_emission is None else carbon_emission

    @electricity_price.setter
    def electricity_price(self, electricity_price: List[float]):
        self.__electricity_price = [np.nan]*self.agent_count if electricity_price is None else electricity_price

    @kwargs.setter
    def kwargs(self, kwargs: Mapping[Any, Any]):
        self.__kwargs = kwargs

    def calculate(self) -> List[float]:
        r"""Calculates default reward.

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{max}(-e_0, 0), \dots, \textrm{max}(-e_n, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        return (np.array(self.electricity_consumption)*-1).clip(max=0).tolist()

    
class MARL(RewardFunction):
    def __init__(self, agent_count: int, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(agent_count, electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        r"""Calculates MARL reward.

        Notes
        -----
        See [1]_ for more information.

        References
        ----------
        .. [1] Vázquez-Canteli, José & Henze, Gregor & Nagy, Zoltán. (2020).
            MARLISA: Multi-Agent Reinforcement Learning with Iterative Sequential
            Action Selection for Load Shaping of Grid-Interactive Connected Buildings. 10.1145/3408308.3427604.
        """

        electricity_consumption = np.array(electricity_consumption)*-1
        total_electricity_consumption = sum(electricity_consumption)
        reward = np.sign(electricity_consumption)*0.01*electricity_consumption**2*np.nanmax(0, sum(total_electricity_consumption))
        return reward.tolist()

class IndependentSACReward(RewardFunction):
    def __init__(self, agent_count: int, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(agent_count, electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        r"""Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

        Recommended for use with the `SAC` controllers.

        Notes
        -----
        Reward value is calculated as :math:`[\textrm{max}(-e_0^3, 0), \dots, \textrm{max}(-e_n^3, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
        """

        electricity_consumption = np.array(self.electricity_consumption)*-1**3
        return electricity_consumption.clip(max=0).tolist()