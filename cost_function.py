import numpy as np

def ramping(net_electric_consumption):
    net_electric_consumption = np.array(net_electric_consumption)
    return np.abs(
        (net_electric_consumption - np.roll(net_electric_consumption,1))[1:]
    ).sum() if len(net_electric_consumption) > 1 else None

def load_factor(net_electric_consumption,minimum_timestep=730):
    net_electric_consumption = np.array(net_electric_consumption)
    timestep = len(net_electric_consumption)
    return np.mean(
        [
            1 - np.mean(net_electric_consumption[i:i+minimum_timestep])/\
                np.max(net_electric_consumption[i:i+minimum_timestep]) 
                for i in range(0,timestep,minimum_timestep)
        ]
    ) if timestep >= minimum_timestep else None

def average_daily_peak(net_electric_consumption,daily_timestep=24):
    net_electric_consumption = np.array(net_electric_consumption)
    timestep = len(net_electric_consumption)
    return np.mean(
        [net_electric_consumption[i:i+daily_timestep].max() for i in range(0,timestep,daily_timestep)]
    ) if timestep >= daily_timestep else None

def peak_demand(net_electric_consumption):
    net_electric_consumption = np.array(net_electric_consumption)
    return net_electric_consumption.max()

def net_electric_consumption(net_electric_consumption):
    net_electric_consumption = np.array(net_electric_consumption)
    return net_electric_consumption.clip(min=0).sum()

def carbon_emissions(carbon_emissions):
    carbon_emissions = np.array(carbon_emissions)
    return carbon_emissions.sum()

def quadratic(net_electric_consumption):
    net_electric_consumption = np.array(net_electric_consumption)
    return (net_electric_consumption.clip(min=0)**2).sum()