import math
import numpy as np
import os
import pandas as pd
from scipy.constants import convert_temperature
from scipy.optimize import fsolve

class BuildingAmericaDomesticHotWater:
    # Reference: https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/house_simulation.pdf
    __DEFAULT_SETPOINT = convert_temperature(130,'Fahrenheit','Celsius')
    __WATER_DENSITY = 988.57 # [kg/m^3]
    __WATER_SPECIFIC_HEAT = 4.186 # [kJ/kg°C]

    def __init__(self,setpoint=None):
        self.setpoint = setpoint

    @property
    def setpoint(self):
        return self.__setpoint

    @property
    def end_use_definitions(self):
        gal_to_m3 = lambda x: x*0.00378541 # [gal] -> [m^3]
        definitions = {
            'clothes_washer': {
                'supply_temperature':self.setpoint,
                'daily_water_use':lambda x: gal_to_m3(2.35 + 0.78*x)
            },
            'dishwasher': {
                'supply_temperature':self.setpoint,
                'daily_water_use':lambda x: gal_to_m3(2.26 + 0.75*x)
            },
            'shower': {
                'supply_temperature':convert_temperature(110,'Fahrenheit','Celsius'),
                'daily_water_use':lambda x: gal_to_m3(14 + 4.67*x)
            },
            'sink': {
                'supply_temperature':convert_temperature(110,'Fahrenheit','Celsius'),
                'daily_water_use':lambda x: gal_to_m3(12.5 + 4.16*x)
            },
        }
        return definitions

    @setpoint.setter
    def setpoint(self,setpoint):
        if setpoint is None:
            self.__setpoint = BuildingAmericaDomesticHotWater.__DEFAULT_SETPOINT
        else:
            self.__setpoint = setpoint

    def get_demand(self,bedrooms,epw_filepath):
        end_use_definitions = self.end_use_definitions
        demand_profile = pd.read_csv(os.path.join(os.path.dirname(__file__),'.misc/building_america_domestic_hot_water.csv'))
        demand_profile['outlet_volume'] = demand_profile.apply(lambda x:
            x['daily_total_volume_fraction']*end_use_definitions[x['end_use']]['daily_water_use'](bedrooms),
            axis=1
        )
        demand_profile['t_mix'] = demand_profile['end_use'].map(lambda x:end_use_definitions[x]['supply_temperature'])
        series = demand_profile.pivot(index='hour',columns='end_use',values=['daily_total_volume_fraction','outlet_volume','t_mix'])
        series.columns = ['-'.join(col).strip() for col in series.columns.values]
        series = series.reset_index()
        t_mains = self.get_mains_temperature(epw_filepath)
        t_mains = pd.DataFrame({'t_mains':t_mains})
        t_mains['day'] = t_mains.index
        t_mains['key'] = 0
        series['key'] = 0
        series = pd.merge(t_mains,series,on='key',how='left')

        def equations(var,*args):
            q_cw, q_dw, q_sh, q_si, v_sh_h, v_si_h, v_sh_c, v_si_c = var
            v_cw, v_dw, v_sh, v_si, t_sh, t_si, t_mains = args
            rho = BuildingAmericaDomesticHotWater.__WATER_DENSITY
            cp = BuildingAmericaDomesticHotWater.__WATER_SPECIFIC_HEAT

            eqs = [
                q_cw - (rho*v_cw*cp*(self.setpoint - t_mains)),
                q_dw - (rho*v_dw*cp*(self.setpoint - t_mains)),
                q_sh - (rho*v_sh*cp*(self.setpoint - t_mains)),
                q_si - (rho*v_si*cp*(self.setpoint - t_mains)),
                v_sh_h*(self.setpoint - t_sh) - v_sh_c*(t_sh - t_mains),
                v_si_h*(self.setpoint - t_si) - v_si_c*(t_si - t_mains),
                v_sh - (v_sh_h + v_sh_c),
                v_si - (v_si_h + v_si_c)
            ]
            return eqs

        var = tuple([1 for _ in range(8)])
        demand = []
    
        for i in range(series.shape[0]):
            args = tuple(series[[
                'outlet_volume-clothes_washer','outlet_volume-dishwasher','outlet_volume-shower','outlet_volume-sink',
                't_mix-shower','t_mix-sink','t_mains',
            ]].iloc[i].values)
            var = fsolve(equations,var,args)
            demand.append(sum(var[0:4])/3600) # [kWh]

        return demand

    @staticmethod
    def get_mains_temperature(epw_filepath):
        df = pd.read_csv(epw_filepath,skiprows=8,header=None)
        df[6] = convert_temperature(df[6],'Celsius','Fahrenheit')
        t_amb_avg = df[6].mean()
        t_ambs = df.groupby(1)[[6]].mean()[6].tolist()
        t_amb_max = max([max(
            [abs(t_ambs[i] - t_ambs[j]) for j in range(i+1,len(t_ambs))]
        ) for i in range(len(t_ambs)-1)])
        offset = 6 # [F]
        ratio = 0.4 + 0.1*(t_amb_avg - 44)
        lag = 35 - 1*(t_amb_avg - 44)
        t_mains = lambda day: (t_amb_avg + offset) + ratio*(t_amb_max/2)*math.sin(math.radians(0.986*(day - 15 - lag) - 90))
        t_mains = [convert_temperature(t_mains(i),'Fahrenheit','Celsius') for i in range(365)]
        return t_mains

class PVSizing:
    def __init__(self,roof_area,demand,climate_zone=2,pv=None):
        self.climate_zone = climate_zone
        self.roof_area = roof_area
        self.demand = demand
        self.pv = pv

    @property
    def climate_zone(self):
        return self.__climate_zone

    @property
    def roof_area(self):
        return self.__roof_area

    @property
    def demand(self):
        return self.__demand

    @property
    def pv(self):
        return self.__pv

    @property
    def default_PV(self):
        '''
        PV Reference:
        SunPower® X-Series Residential Solar Panels X21-345,
        https://us.sunpower.com/sites/default/files/media-library/data-sheets/ds-x21-series-335-345-residential-solar-panels.pdf
        '''
        return {
            'width':1.046, # [m]
            'length':1.558, # [m]
            'efficiency':0.215,
            'rating':0.250, # [kW]
        }

    @climate_zone.setter
    def climate_zone(self,climate_zone):
        self.__climate_zone = climate_zone

    @roof_area.setter
    def roof_area(self,roof_area):
        self.__roof_area = roof_area

    @demand.setter
    def demand(self,demand):
        self.__demand = demand

    @pv.setter
    def pv(self,pv):
        if pv is None:
            self.__pv = self.default_PV
        else:
            self.__pv = {**self.default_PV,**pv}

    def size(self,method='peak_demand',peak_min_hour=0,peak_max_hour=24):
        methods = ['peak_demand','peak_count','annual_demand']
        pv_count_limit = math.floor(self.roof_area/(self.pv['length']*self.pv['width']))
        data = pd.read_csv(os.path.join(os.path.dirname(__file__),'.misc/solar_generation_1kW.csv'))
        data = data[data['climate_zone']==self.climate_zone].copy()
        data['timestamp'] = pd.date_range('2019-01-01 00:00:00','2019-12-31 23:00:00',freq='H')
        data['hour'] = data['timestamp'].dt.hour
        data['ac_inverter_power_per_kw'] /= 1000 # [W] -> [kW]
        data['adjusted_rating'] = self.pv['rating']*data['ac_inverter_power_per_kw']
        data['demand'] = self.demand
        data['pv_count'] = np.ceil(data['demand']/data['adjusted_rating'])
        data = data[(data['pv_count'] < np.inf)].copy()

        if method == 'peak_demand':
            pv_count = data[
                (data['hour']>=peak_min_hour)
                &(data['hour']<=peak_max_hour)
            ].nlargest(1,'demand').iloc[0]['pv_count']

        elif method == 'peak_count':
            pv_count = data[
                (data['hour']>=peak_min_hour)
                &(data['hour']<=peak_max_hour)
            ]['pv_count'].max()

        elif method == 'annual_demand':
            pv_count = math.ceil(data['demand'].sum()/data['adjusted_rating'].sum())

        else:
            raise ValueError(f'Invalid method. Valid methods are {methods}.')

        return pv_count, pv_count_limit