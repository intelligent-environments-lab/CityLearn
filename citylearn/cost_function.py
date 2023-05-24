from typing import List, Tuple
import numpy as np
import pandas as pd

class CostFunction:
    r"""Cost and energy flexibility functions that may be used to evaluate environment performance."""

    @staticmethod
    def ramping(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of absolute difference in net electric consumption between consecutive time steps.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.

        Returns
        -------
        ramping : List[float]
            Ramping cost.

        Notes
        -----        
        .. math:: 
            \textrm{ramping} = \sum_{i=1}^{n}{\lvert E_i - E_{i-1} \rvert}
            
        Where :math:`E_i` is the :math:`i^{\textrm{th}}` element in `net_electricity_consumption`, :math:`E`, that has a length of :math:`n`.
        """

        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['ramping'] = data['net_electricity_consumption'] - data['net_electricity_consumption'].shift(1)
        data['ramping'] = data['ramping'].abs()
        data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['ramping'].tolist()

    @staticmethod
    def one_minus_load_factor(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Difference between 1 and the load factor i.e., ratio of rolling mean demand 
        to rolling peak demand over a specified period.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 730
            Period window/time steps.

        Returns
        -------
        1 - load_factor : List[float]
            1 - load factor cost.
        """

        window = 730 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['group'] = (data.index/window).astype(int)
        data = data.groupby(['group'])[['net_electricity_consumption']].agg(['mean','max'])
        data['load_factor'] = 1 - (data[('net_electricity_consumption','mean')]/data[('net_electricity_consumption','max')])
        data['load_factor'] = data['load_factor'].rolling(window=data.shape[0],min_periods=1).mean()
        
        return data['load_factor'].tolist()

    @staticmethod
    def peak(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Net electricity consumption peak.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 24
            Period window/time steps to find peaks.
            
        Returns
        -------
        peak : List[float]
            Average daily peak cost.
        """

        window = 24 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['group'] = (data.index/window).astype(int)
        data = data.groupby(['group'])[['net_electricity_consumption']].max()
        data['net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).mean()
        
        return data['net_electricity_consumption'].tolist()

    @staticmethod
    def electricity_consumption(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of positive electricity consumption.

        It is the sum of electricity that is consumed from the grid.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        electricity_consumption : List[float]
            Electricity consumption cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['electricity_consumption'].tolist()

    @staticmethod
    def zero_net_energy(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of net electricity consumption.

        It is the net sum of electricty that is consumed from the grid and self-generated from renenewable sources.
        This calculation of zero net energy does not consider TDV and all time steps are weighted equally.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        zero_net_energy : List[float]
            Zero net energy cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption)})
        data['zero_net_energy'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['zero_net_energy'].tolist()

    @staticmethod
    def carbon_emissions(carbon_emissions: List[float]) -> List[float]:
        r"""Rolling sum of carbon emissions.

        Parameters
        ----------
        carbon_emissions : List[float]
            Carbon emissions time series.
            
        Returns
        -------
        carbon_emissions : List[float]
            Carbon emissions cost.
        """

        data = pd.DataFrame({'carbon_emissions':np.array(carbon_emissions).clip(min=0)})
        data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['carbon_emissions'].tolist()

    @staticmethod
    def cost(cost: List[float]) -> List[float]:
        r"""Rolling sum of electricity monetary cost.

        Parameters
        ----------
        cost : List[float]
            Cost time series.
            
        Returns
        -------
        cost : List[float]
            Cost of electricity.
        """

        data = pd.DataFrame({'cost':np.array(cost).clip(min=0)})
        data['cost'] = data['cost'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['cost'].tolist()

    @staticmethod
    def quadratic(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of net electricity consumption raised to the power of 2.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        quadratic : List[float]
            Quadratic cost.

        Notes
        -----
        Net electricity consumption values are clipped at a minimum of 0 before calculating the quadratic cost.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['quadratic'] = data['net_electricity_consumption']**2
        data['quadratic'] = data['quadratic'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['quadratic'].tolist()
    
    @staticmethod
    def discomfort(indoor_dry_bulb_temperature: List[float], dry_bulb_temperature_set_point: List[float], band: float = None, occupant_count: List[int] = None) -> Tuple[list]:
        r"""Rolling percentage of discomfort (total, too cold, and too hot) time steps as well as rolling minimum, maximum and average temperature delta.

        Parameters
        ----------
        indoor_dry_bulb_temperature: List[float]
            Average building dry bulb temperature time series.
        dry_bulb_temperature_set_point: List[float]
            Building thermostat setpoint time series.
        band: float, default = 2.0
            Comfort band above and below dry_bulb_temperature_set_point beyond 
            which occupant is assumed to be uncomfortable.
        occupant_cunt: List[float], optional
            Occupant count time series. If provided, the comfort cost is 
            evaluated for occupied time steps only.
            
        Returns
        -------
        discomfort: List[float]
            Rolling proportion of occupied timesteps where the condition 
            (dry_bulb_temperature_set_point - band) <= indoor_dry_bulb_temperature <= (dry_bulb_temperature_set_point + band) is not met.
        discomfort_too_cold: List[float]
            Rolling proportion of occupied timesteps where the condition indoor_dry_bulb_temperature < (dry_bulb_temperature_set_point - band) is met.
        discomfort_too_hot: List[float]
            Rolling proportion of occupied timesteps where the condition indoor_dry_bulb_temperature > (dry_bulb_temperature_set_point + band) is met.
        discomfort_delta_minimum: List[float]
            Rolling minimum of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        discomfort_delta_maximum: List[float]
            Rolling maximum of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        discomfort_delta_average: List[float]
            Rolling average of indoor_dry_bulb_temperature - dry_bulb_temperature_set_point.
        """

        band = 2.0 if band is None else band

        # unmet hours
        data = pd.DataFrame({
            'indoor_dry_bulb_temperature': indoor_dry_bulb_temperature, 
            'dry_bulb_temperature_set_point': dry_bulb_temperature_set_point,
            'occupant_count': [1]*len(indoor_dry_bulb_temperature) if occupant_count is None else occupant_count
        })
        occupied_time_step_count = data[data['occupant_count'] > 0.0].shape[0]
        data['delta'] = data['indoor_dry_bulb_temperature'] - data['dry_bulb_temperature_set_point']
        data.loc[data['occupant_count'] == 0.0, 'delta'] = 0.0
        data['discomfort'] = 0
        data.loc[data['delta'].abs() > band, 'discomfort'] = 1
        data['discomfort'] = data['discomfort'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # too cold
        data['discomfort_too_cold'] = 0
        data.loc[data['delta'] < -band, 'discomfort_too_cold'] = 1
        data['discomfort_too_cold'] = data['discomfort_too_cold'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # too hot
        data['discomfort_too_hot'] = 0
        data.loc[data['delta'] > band, 'discomfort_too_hot'] = 1
        data['discomfort_too_hot'] = data['discomfort_too_hot'].rolling(window=data.shape[0],min_periods=1).sum()/occupied_time_step_count

        # minimum delta
        data['discomfort_delta_minimum'] = data['delta'].rolling(window=data.shape[0],min_periods=1).min()

        # maximum delta
        data['discomfort_delta_maximum'] = data['delta'].rolling(window=data.shape[0],min_periods=1).max()

        # average delta
        data['discomfort_delta_average'] = data['delta'].rolling(window=data.shape[0],min_periods=1).mean()

        return (
            data['discomfort'].tolist(),
            data['discomfort_too_cold'].tolist(),
            data['discomfort_too_hot'].tolist(),
            data['discomfort_delta_minimum'].tolist(),
            data['discomfort_delta_maximum'].tolist(),
            data['discomfort_delta_average'].tolist()
        )
        