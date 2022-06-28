from typing import List
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

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0]
        >>> CostFunction.ramping(net_electricity_consumption)
        [nan, 100.0, 100.0, 500.0, 700.0]
        """

        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['ramping'] = data['net_electricity_consumption'] - data['net_electricity_consumption'].shift(1)
        data['ramping'] = data['ramping'].abs()
        data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['ramping'].tolist()

    @staticmethod
    def load_factor(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Difference between 1 and the rolling mean ratio of rolling mean demand to rolling peak demand over a specified period.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 730
            Period window/time steps.

        Returns
        -------
        load_factor : List[float]
            Load factor cost.

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0]
        >>> window = 2
        >>> CostFunction.load_factor(net_electricity_consumption, window=window)
        [nan, 0.25, 0.125, 0.19444444444444445, 0.1875]
        """

        window = 730 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['mean_net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=window,min_periods=window).mean()
        data['max_net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=window,min_periods=window).max()
        data['load_factor'] = 1 - (data['mean_net_electricity_consumption']/data['max_net_electricity_consumption'])
        data['load_factor'] = data['load_factor'].rolling(window=data.shape[0],min_periods=1).mean()
        return data['load_factor'].tolist()

    @staticmethod
    def average_daily_peak(net_electricity_consumption: List[float], daily_time_step: int = None) -> List[float]:
        r"""Rolling mean of daily net electricity consumption peaks.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        daily_time_step : int, default: 24
            Number of time steps in a day.
            
        Returns
        -------
        average_daily_peak : List[float]
            Average daily peak cost.

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> daily_time_step = 2
        >>> CostFunction.average_daily_peak(net_electricity_consumption, daily_time_step=daily_time_step)
        [nan, 200.0, 200.0, 333.3333333333333, 400.0, 400.0]
        """

        daily_time_step = 24 if daily_time_step is None else daily_time_step
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['average_daily_peak'] = data['net_electricity_consumption'].rolling(window=daily_time_step,min_periods=daily_time_step).max()
        data['average_daily_peak'] = data['average_daily_peak'].rolling(window=data.shape[0],min_periods=1).mean()
        return data['average_daily_peak'].tolist()

    @staticmethod
    def peak_demand(net_electricity_consumption: List[float], window: int = None) -> List[float]:
        r"""Net electricity consumption peaks.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
        window : int, default: 8760
            Period window/time steps to find peaks.
            
        Returns
        -------
        peak_demand : List[float]
            Peak demand cost.        

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> window = 2
        >>> CostFunction.peak_demand(net_electricity_consumption, window=window)
        [nan, 200.0, 200.0, 600.0, 600.0, 400.0]
        """

        window = 8760 if window is None else window
        data = pd.DataFrame({'net_electricity_consumption':net_electricity_consumption})
        data['peak_demand'] = data['net_electricity_consumption'].rolling(window=window,min_periods=window).max()
        return data['peak_demand'].tolist()

    @staticmethod
    def net_electricity_consumption(net_electricity_consumption: List[float]) -> List[float]:
        r"""Rolling sum of net electricity consumption.

        Parameters
        ----------
        net_electricity_consumption : List[float]
            Electricity consumption time series.
            
        Returns
        -------
        net_electricity_consumption : List[float]
            Net electricity consumption cost.        

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> CostFunction.net_electricity_consumption(net_electricity_consumption)
        [100.0, 300.0, 500.0, 1100.0, 1500.0, 1800.0]
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['net_electricity_consumption'].tolist()

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

        Examples
        --------
        >>> carbon_emissions = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> CostFunction.carbon_emissions(carbon_emissions)
        [100.0, 300.0, 500.0, 1100.0, 1500.0, 1800.0]
        """

        data = pd.DataFrame({'carbon_emissions':carbon_emissions})
        data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['carbon_emissions'].tolist()

    @staticmethod
    def price(price: List[float]) -> List[float]:
        r"""Rolling sum of electricity price.

        Parameters
        ----------
        price : List[float]
            Price time series.
            
        Returns
        -------
        price : List[float]
            Price cost.        

        Examples
        --------
        >>> price = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> CostFunction.price(carbon_emissions)
        [100.0, 300.0, 500.0, 1100.0, 1500.0, 1800.0]
        """

        data = pd.DataFrame({'price':np.array(price).clip(min=0)})
        data['price'] = data['price'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['price'].tolist()

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

        Examples
        --------
        >>> net_electricity_consumption = [100.0, 200.0, 200.0, 600.0, 400.0, 300.0]
        >>> CostFunction.quadratic(net_electricity_consumption)
        [10000.0, 50000.0, 90000.0, 450000.0, 610000.0, 700000.0]
        """

        data = pd.DataFrame({'net_electricity_consumption':np.array(net_electricity_consumption).clip(min=0)})
        data['quadratic'] = data['net_electricity_consumption']**2
        data['quadratic'] = data['quadratic'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['quadratic'].tolist()