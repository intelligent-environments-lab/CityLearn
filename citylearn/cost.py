from typing import List
import numpy as np
import pandas as pd

class Function:
    @staticmethod
    def ramping(net_electric_consumption: List[float]) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
        data['ramping'] = data['net_electric_consumption'] - data['net_electric_consumption'].shift(1)
        data['ramping'] = data['ramping'].abs()
        data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['ramping'].tolist()

    @staticmethod
    def load_factor(net_electric_consumption: List[float], minimum_time_step: int = 730) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
        data['mean_net_electric_consumption'] = data['net_electric_consumption'].rolling(window=minimum_time_step,min_periods=minimum_time_step).mean()
        data['max_net_electric_consumption'] = data['net_electric_consumption'].rolling(window=minimum_time_step,min_periods=minimum_time_step).max()
        data['load_factor'] = 1 - (data['mean_net_electric_consumption']/data['max_net_electric_consumption'])
        data['load_factor'] = data['load_factor'].rolling(window=data.shape[0],min_periods=1).mean()
        return data['load_factor'].tolist()

    @staticmethod
    def average_daily_peak(net_electric_consumption: List[float], daily_time_step: int = 24) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
        data['average_daily_peak'] = data['net_electric_consumption'].rolling(window=daily_time_step,min_periods=daily_time_step).max()
        data['average_daily_peak'] = data['average_daily_peak'].rolling(window=data.shape[0],min_periods=1).mean()
        return data['average_daily_peak'].tolist()

    @staticmethod
    def peak_demand(net_electric_consumption: List[float], minimum_timestep: int = 8760) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
        data['peak_demand'] = data['net_electric_consumption'].rolling(window=minimum_timestep,min_periods=minimum_timestep).max()
        return data['peak_demand'].tolist()

    @staticmethod
    def net_electric_consumption(net_electric_consumption: List[float]) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':np.array(net_electric_consumption).clip(min=0)})
        data['net_electric_consumption'] = data['net_electric_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['net_electric_consumption'].tolist()

    @staticmethod
    def carbon_emissions(carbon_emissions: List[float]) -> List[float]:
        data = pd.DataFrame({'carbon_emissions':carbon_emissions})
        data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['carbon_emissions'].tolist()

    @staticmethod
    def quadratic(net_electric_consumption: List[float]) -> List[float]:
        data = pd.DataFrame({'net_electric_consumption':np.array(net_electric_consumption).clip(min=0)})
        data['quadratic'] = data['net_electric_consumption']**2
        data['quadratic'] = data['quadratic'].rolling(window=data.shape[0],min_periods=1).sum()
        return data['quadratic'].tolist()