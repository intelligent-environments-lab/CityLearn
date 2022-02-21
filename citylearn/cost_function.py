import numpy as np
import pandas as pd

def ramping(net_electric_consumption):
    data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
    data['ramping'] = data['net_electric_consumption'] - data['net_electric_consumption'].shift(1)
    data['ramping'] = data['ramping'].abs()
    data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()
    return data['ramping'].tolist()

def load_factor(net_electric_consumption,minimum_timestep=730):
    data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
    data['mean_net_electric_consumption'] = data['net_electric_consumption'].rolling(window=minimum_timestep,min_periods=minimum_timestep).mean()
    data['max_net_electric_consumption'] = data['net_electric_consumption'].rolling(window=minimum_timestep,min_periods=minimum_timestep).max()
    data['load_factor'] = 1 - (data['mean_net_electric_consumption']/data['max_net_electric_consumption'])
    data['load_factor'] = data['load_factor'].rolling(window=data.shape[0],min_periods=1).mean()
    return data['load_factor'].tolist()

def average_daily_peak(net_electric_consumption,daily_timestep=24):
    data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
    data['average_daily_peak'] = data['net_electric_consumption'].rolling(window=daily_timestep,min_periods=daily_timestep).max()
    data['average_daily_peak'] = data['average_daily_peak'].rolling(window=data.shape[0],min_periods=1).mean()
    return data['average_daily_peak'].tolist()

def peak_demand(net_electric_consumption,minimum_timestep=8760):
    data = pd.DataFrame({'net_electric_consumption':net_electric_consumption})
    data['peak_demand'] = data['net_electric_consumption'].rolling(window=minimum_timestep,min_periods=minimum_timestep).max()
    return data['peak_demand'].tolist()

def net_electric_consumption(net_electric_consumption):
    data = pd.DataFrame({'net_electric_consumption':np.array(net_electric_consumption).clip(min=0)})
    data['net_electric_consumption'] = data['net_electric_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
    return data['net_electric_consumption'].tolist()

def carbon_emissions(carbon_emissions):
    data = pd.DataFrame({'carbon_emissions':carbon_emissions})
    data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
    return data['carbon_emissions'].tolist()

def quadratic(net_electric_consumption):
    data = pd.DataFrame({'net_electric_consumption':np.array(net_electric_consumption).clip(min=0)})
    data['quadratic'] = data['net_electric_consumption']**2
    data['quadratic'] = data['quadratic'].rolling(window=data.shape[0],min_periods=1).sum()
    return data['quadratic'].tolist()