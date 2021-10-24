from math import radians, sin
import pandas as pd
from scipy.constants import convert_temperature
from scipy.optimize import fsolve

class BuildingAmericaDomesticHotWater:
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
        # Reference: https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/house_simulation.pdf

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
        demand_profile = pd.read_csv('.misc/building_america_domestic_hot_water.csv')
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
        # Reference: https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/house_simulation.pdf

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
        t_mains = lambda day: (t_amb_avg + offset) + ratio*(t_amb_max/2)*sin(radians(0.986*(day - 15 - lag) - 90))
        t_mains = [convert_temperature(t_mains(i),'Fahrenheit','Celsius') for i in range(365)]
        return t_mains