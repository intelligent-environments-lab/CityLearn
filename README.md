# CityLearn
CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. Its objective is to facilitiate and standardize the evaluation of RL agents such that different algorithms can be easily compared with each other.
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/dr.jpg)
## Description
Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. Demand response is the coordination of electricity consuming agents (i.e. buildings) in order to reshape the overall curve of electrical demand.
CityLearn allows the easy implementation of reinforcement learning agents in a multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent. Currently, CityLearn allows controlling the storage of domestic hot water (DHW), and chilled water (for sensible cooling and dehumidification). CityLearn also includes models of air-to-water heat pumps, electric heaters, solar photovoltaic arrays, and the pre-computed energy loads of the buildings, which include space cooling, dehumidification, appliances, DHW, and solar generation.
## Dependencies
CityLearn requires the installation of the following Python libraries:
- Pandas 0.24.2 or older
- Numpy 1.16.4 or older
- Gym 0.14.0
- PyTorch 1.1.0
- Json 2.0.9

CityLearn may still work with some earlier versions of these libraries, but we have tested it with those.

## Files
- [main.ipynb](/main.ipynb): jupyter lab file. Example of the implementation of a reinforcement learning agent ([TD3](https://arxiv.org/abs/1802.09477)) in a single building in ```CityLearn```
- [main.py](/main.py): Copy of [main.ipynb](/main.ipynb) as a python file.
- [buildings_state_action_space.json](/buildings_state_action_space.json): json file containing the possible states and actions for every building, from which users can choose.
- [building_attributes.json](/data/building_attributes.json): json file containing the attributes of the buildings and which users can modify.
- [citylearn.py](/citylearn.py): Contains the ```CityLearn``` environment and the functions ```building_loader()``` and ```autosize()```
- [energy_models.py](/energy_models.py): Contains the classes ```Building```, ```HeatPump``` and ```EnergyStorage```, which are called by the ```CityLearn``` class.
- [agent.py](/agent.py): Implementation of the TD3 algorithm ([TD3](https://arxiv.org/abs/1802.09477)) RL algorithm. This file must be modified with any other RL implementation, which can then be run in the [main.ipynb](/main.ipynb) jupyter lab file or the [main.py](/main.py) file.
- [reward_function.py](/reward_function.py): Contains the reward function that wraps and modifies the rewards obtained from ```CityLearn```. This function can be modified by the user in order to minimize the cost function of ```CityLearn```.
- [example_rbc.ipynb](/example_rbc.ipynb): jupyter lab file. Example of the implementation of a manually optimized Rule-based controller (RBC) that can be used for comparison
### Classes
- CityLearn
  - Building
    - HeatPump
    - ElectricHeater
    - EnergyStorage
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/citylearn_diagram.jpg)

### CityLearn
This class of type OpenAI Gym Environment contains all the buildings and their subclasses.
- CityLearn attributes (all in kWh)
  - ```net_electric_consumption```: district net electricity consumption
  - ```net_electric_consumption_no_storage```: district net electricity consumption if there were no cooling and DHW storage
  - ```net_electric_consumption_no_pv_no_storage```: district net electricity consumption if there were no cooling, DHW storage and PV generation
  - ```electric_consumption_dhw_storage```: electricity consumed in the district to increase DHW energy storage (when > 0) and electricity that the decrease in DHW energy storage saves from consuming in the district (when < 0).
  - ```electric_consumption_cooling_storage```: electricity consumed in the district to increase cooling energy storage (when > 0) and electricity that the decrease in cooling energy storage saves from consuming in the district (when < 0).
  - ```electric_consumption_dhw```: electricity consumed to satisfy the DHW demand of the district
  - ```electric_consumption_cooling```: electricity consumed to satisfy the cooling demand of the district
  - ```electric_consumption_appliances```: non-shiftable electricity consumed by appliances
  - ```electric_generation```: electricity generated in the district 
- CityLearn specific methods
  - ```get_state_action_spaces()```: returns state-action spaces for all the buildings
  - ```next_hour()```: advances simulation to the next time-step
  - ```get_building_information()```: returns attributes of the buildings that can be used by the RL agents (i.e. to implement building-specific RL agents based on their attributes, or control buildings with correlated demand profiles by the same agent)
  - ```get_baseline_cost()```: returns the costs of a Rule-based controller (RBC), which is used to divide the final cost by it.
  - ```cost()```: returns the normlized cost of the enviornment after it has been simulated. cost < 1 when the controller's performance is better than the RBC.
- Methods inherited from OpenAI Gym
  - ```step()```: advances simulation to the next time-step and takes an action based on the current state
  - ```_get_ob()```: returns all the states
  - ```_terminal()```: returns True if the simulation has ended
  - ```seed()```: specifies a random seed

### Building
The DHW and cooling demands of the buildings have been pre-computed and obtained from EnergyPlus. The DHW and cooling supply systems are sized such that the DHW and cooling demands are always satisfied. CityLearn automatically sets constraints to the actions from the controllers to guarantee that the DHW and cooling demands are satisfied, and that the building does not receive from the storage units more energy than it needs. 
The file building_attributes.json contains the attributes of each building, which can be modified. We do not advise to modify the attributes Building -> HeatPump -> nominal_power and Building -> ElectricHeater -> nominal_power from their default value "autosize", as they guarantee that the DHW and cooling demand are always satisfied.
- Building attributes (all in kWh)
  - ```cooling_demand_building```: demand for cooling energy to cool down and dehumidify the building
  - ```dhw_demand_building```: demand for heat to supply the building with domestic hot water (DHW)
  - ```electric_consumption_appliances```: non-shiftable electricity consumed by appliances
  - ```electric_generation```: electricity generated by the solar panels
  - ```electric_consumption_cooling```: electricity consumed to satisfy the cooling demand of the building
  - ```electric_consumption_cooling_storage```: electricity consumed by the buildin heat pump to increase cooling energy storage (when > 0) and electricity that the decrease in cooling energy storage saves from consuming in the building (when < 0).
  - ```electric_consumption_dhw```: electricity consumed to satisfy the DHW demand of the building
  - ```electric_consumption_dhw_storage```: electricity consumed by the building heat pump to increase DHW energy storage (when > 0) and electricity that the decrease in DHW energy storage saves from consuming in the building (when < 0).
  - ```net_electric_consumption```: building net electricity consumption
  - ```net_electric_consumption_no_storage```: building net electricity consumption if there were no cooling and DHW storage
  - ```net_electric_consumption_no_pv_no_storage```: building net electricity consumption if there were no cooling, DHW storage and PV generation
  - ```cooling_device_to_building```: cooling energy supplied by the cooling device (heat pump) to the building
  - ```cooling_storage_to_building```: cooling energy supplied by the cooling storage device (chilled water tank) to the building
  - ```cooling_device_to_storage```: cooling energy supplied by the cooling device to the cooling storage device
  - ```cooling_storage_soc```: state of charge of the cooling storage device
  - ```dhw_heating_device_to_building```: DHW energy supplied by the heating device to the building
  - ```dhw_storage_to_building```: DHW energy supplied by the DHW storage device to the building
  - ```dhw_heating_device_to_storage```: DHW energy supplied by the heating device to the DHW storage device
  - ```dhw_storage_soc```: state of charge of the DHW storage device

- Methods
  - ```set_state_space()``` and ```set_action_space()``` set the state-action space of each building
  - ```set_storage_heating()``` and ```set_storage_cooling()``` set the state of charge of the ```EnergyStorage``` device to the specified value and within the physical constraints of the system. Returns the total electricity consumption of the building at that time-step.
  - ```get_non_shiftable_load()```, ```get_solar_power()```, ```get_dhw_electric_demand()``` and ```get_cooling_electric_demand()``` get the different types of electricity demand and generation.
  
### Heat pump
Its efficiency is given by the coefficient of performance (COP), which is calculated as a function of the outdoor air temperature and of the following parameters:

-```eta_tech```: technical efficiency of the heat pump

-```T_target```: target temperature. Conceptually, it is  equal to the logarithmic mean of the temperature of the supply water of the storage device and the temperature of the water returning from the building. Here it is assumed to be constant and defined by the user in the [building_attributes.json](/data/building_attributes.json) file.  For cooling, values between 7C and 10C are reasonable.
Any amount of cooling demand of the building that isn't satisfied by the ```EnergyStorage``` device is automatically supplied by the ```HeatPump``` directly to the ```Building```, guaranteeing that the cooling demand is always satisfied. The ```HeatPump``` is more efficient (has a higher COP) if the outdoor air temperature is lower, and less efficient (lower COP) when the outdoor temperature is higher (typically during the day time). On the other hand, the electricity demand is typically higher during the daytime and lower at night. ```cooling_energy_generated = COP*electricity_consumed, COP > 1```
- Attributes
  - ```cop_heating```: coefficient of performance for heating supply
  - ```cop_cooling```:  coefficient of performance for cooling supply
  - ```electrical_consumption_cooling```: electricity consumed for heating supply (kWh)
  - ```electrical_consumption_heating```: electricity consumed for cooling supply (kWh)
  - ```heat_supply```: heating supply (kWh)
  - ```cooling_supply```: cooling supply (kWh)
- Methods
  - ```get_max_cooling_power()``` and ```get_max_heating_power()``` compute the maximum amount of heating or cooling that the heat pump can provide based on its nominal power of the compressor and its COP. 
  - ```get_electric_consumption_cooling()``` and ```get_electric_consumption_heating()``` return the amount of electricity consumed by the heat pump for a given amount of supplied heating or cooling energy.
### Energy storage
Storage devices allow heat pumps to store energy that can be later released into the building. Typically every building will have its own storage device, but CityLearn also allows defining a single instance of the ```EnergyStorage``` for multiple instances of the class ```Building```, therefore having a group of buildings sharing a same energy storage device.

- Attributes
  - ```soc```: state of charge (kWh)
  - ```energy_balance```: energy coming in (if positive) or out (if negative) of the energy storage device (kWh)

- Methods
  - ```charge()``` increases (+) or decreases (-) of the amount of energy stored. The input is the amount of energy as a ratio of the total capacity of the storage device (can vary from -1 to 1). Outputs the energy balance of the storage device.
## Environment variables
The file [buildings_state_action_space.json](/buildings_state_action_space.json) contains all the states and action variables that the buildings can possibly return:
### Possible states
- ```month```: 1 (January) through 12 (December)
- ```day```: type of day as provided by EnergyPlus (from 1 to 8). 1 (Sunday), 2 (Monday), ..., 7 (Saturday), 8 (Holiday)
- ```hour```: hour of day (from 1 to 24).
- ```daylight_savings_status```: indicates if the building is under daylight savings period (0 to 1). 0 indicates that the building has not changed its electricity consumption profiles due to daylight savings, while 1 indicates the period in which the building may have been affected.
- ```t_out```: outdoor temperature in Celcius degrees.
- ```t_out_pred_6h```: outdoor temperature predicted 6h ahead (accuracy: +- )
- ```t_out_pred_12h```: outdoor temperature predicted 12h ahead (accuracy: +- )
- ```t_out_pred_24h```: outdoor temperature predicted 24h ahead (accuracy: +- )
- ```rh_out```: outdoor relative humidity in %.
- ```rh_out_pred_6h```: outdoor relative humidity predicted 6h ahead (accuracy: +- )
- ```rh_out_pred_12h```: outdoor relative humidity predicted 12h ahead (accuracy: +- )
- ```rh_out_pred_24h```: outdoor relative humidity predicted 24h ahead (accuracy: +- )
- ```diffuse_solar_rad```: diffuse solar radiation in W/m^2.
- ```diffuse_solar_rad_pred_6h```: diffuse solar radiation predicted 6h ahead (accuracy: +- )
- ```diffuse_solar_rad_pred_12h```: diffuse solar radiation predicted 12h ahead (accuracy: +- )
- ```diffuse_solar_rad_pred_24h```: diffuse solar radiation predicted 24h ahead (accuracy: +- )
- ```direct_solar_rad```: direct solar radiation in W/m^2.
- ```direct_solar_rad_pred_6h```: direct solar radiation predicted 6h ahead (accuracy: +- )
- ```direct_solar_rad_pred_12h```: direct solar radiation predicted 12h ahead (accuracy: +- )
- ```direct_solar_rad_pred_24h```: direct solar radiation predicted 24h ahead (accuracy: +- )
- ```t_in```: indoor temperature in Celcius degrees.
- ```avg_unmet_setpoint```: average difference between the indoor temperatures and the cooling temperature setpoints in the different zones of the building in Celcius degrees. sum((t_in - t_setpoint).clip(min=0) * zone_volumes)/total_volume
- ```rh_in```: indoor relative humidity in %.
- ```non_shiftable_load```: electricity currently consumed by electrical appliances in kWh.
- ```solar_gen```: electricity currently being generated by photovoltaic panels in kWh.
- ```cooling_storage_soc```: state of the charge (SOC) of the cooling storage device. From 0 (no energy stored) to 1 (at full capacity).
- ```dhw_storage_soc```: state of the charge (SOC) of the domestic hot water (DHW) storage device. From 0 (no energy stored) to 1 (at full capacity).

### Possible actions
- ```cooling_storage```: increase (action > 0) or decrease (action < 0) of the amount of cooling energy stored in the cooling storage device. -1.0 <= action <= 1.0 (attempts to decrease or increase the cooling energy stored in the storage device by an amount equal to the action times the storage device's maximum capacity). In order to decrease the energy stored in the device (action < 0), the energy must be released into the building. Therefore, the state of charge will not decrease proportionally to the action taken if the demand for cooling of the building is lower than the action times the maximum capacity of the cooling storage device.
- ```dhw_storage```: increase (action > 0) or decrease (action < 0) of the amount of DHW stored in the DHW storage device. -1.0 <= action <= 1.0 (attempts to decrease or increase the DHW stored in the storage device by an amount equivalent to action times its maximum capacity). In order to decrease the energy stored in the device, the energy must be released into the building. Therefore, the state of charge will not decrease proportionally to the action taken if the demand for DHW of the building is lower than the action times the maximum capacity of the DHW storage device.

The mathematics of this control logic can be seen in the methods ```set_storage_heating(action)``` and ```set_storage_cooling(action)``` of the class Building in the file [energy_models.py](/energy_models.py).
### Reward function
- ```r```: the reward returned by CityLearn is the electricity consumption of every building for a given hour. The function ```reward_function``` can be used to convert ```r``` into the final reward that the RL agent will receive. ```reward_function.py``` contains the function ```reward_function```, which should be modified in a way that can allows the agent to minimize the selected cost function of the environment.
### Cost function
```env.cost()``` is the cost function of the environment, which the RL controller must minimize. There are multiple cost functions available, which are all defined as a function of the total non-negative net electricity consumption of the whole neighborhood:
- ```ramping```: sum(|e(t)-e(t-1)|), where e is the net non-negative electricity consumption every time-step.
- ```1-load_factor```: the load factor is the average net electricity load divided by the maximum electricity load.
- ```average_daily_peak```: average daily peak net demand.
- ```peak_demand```: maximum peak electricity demand
- ```net_electricity_consumption```: total amount of electricity consumed
- ```quadratic```: sum(e^2), where e is the net non-negative electricity consumption every time-step. (Not used in The CityLearn Challenge).

## Additional functions
- ```building_loader(demand_file, weather_file, buildings)``` receives a dictionary with all the building instances and their respectives IDs, and loads them with the data of heating and cooling loads from the simulations.
- ```auto_size(buildings, t_target_heating, t_target_cooling)``` automatically sizes the heat pumps and the storage devices. It assumes fixed target temperatures of the heat pump for heating and cooling, which combines with weather data to estimate their hourly COP for the simulated period. The ```HeatPump``` is sized such that it will always be able to fully satisfy the heating and cooling demands of the building. This function also sizes the ```EnergyStorage``` devices, setting their capacity as 3 times the maximum hourly cooling demand in the simulated period.
## Multi-agent coordination
### One building
  - A good control policy for cooling is trivial, and consists on storing cooling energy during the night (when the cooling demand of the building is low and the COP of the heat pump is higher), and releasing the stored cooling energy into the building during the day (high demand for cooling and low COP). 
### Multiple buildings
  - If controlled independently of each other and without coordination or sharing any information, the buildings will tend to consume more electricity simultaneously, which may not be optimal if the objective is peak reduction and load flattening. 
### [Challenge](https://sites.google.com/view/citylearnchallenge)
Coordinate multiple RL agents or a single centralized agent to control all the buildings. The agents may share certain information with each other. The objective is to reduce the cost function by smoothening, reducing, and flattening the total net demand for electricity of the whole district. Check out our [CityLearn Challenge](https://sites.google.com/view/citylearnchallenge)
## Cite CityLearn
- [Vázquez-Canteli, J.R., Kämpf, J., Henze, G., and Nagy, Z., "CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep Reinforcement Learning", Proceedings of the 6th ACM International Conference, ACM New York p. 356-357, New York, 2019](https://dl.acm.org/citation.cfm?id=3360998)
## Related Publications
- [Vázquez-Canteli, J.R., and Nagy, Z., “Reinforcement Learning for Demand Response: A Review of algorithms and modeling techniques”, Applied Energy 235, 1072-1089, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S0306261918317082)
- [Vázquez-Canteli, J.R., Ulyanin, S., Kämpf J., and Nagy, Z., “Fusing TensorFlow with building energy simulation for intelligent energy management in smart cities”, Sustainable Cities and Society, 2018.](https://www.sciencedirect.com/science/article/abs/pii/S2210670718314380)
- [Vázquez-Canteli J.R., Kämpf J., and Nagy, Z., “Balancing comfort and energy consumption of a heat pump using batch reinforcement learning with fitted Q-iteration”, CISBAT, Lausanne, 2017](https://www.sciencedirect.com/science/article/pii/S1876610217332629)
## Contact
- Email: citylearn@utexas.edu
- [José R. Vázquez-Canteli](https://www.researchgate.net/profile/Jose_Vazquez-Canteli2), PhD Candidate at The University of Texas at Austin, Department of Civil, Architectural, and Environmental Engineering. [Intelligent Environments Laboratory (IEL)](https://nagy.caee.utexas.edu/). 
- [Dr. Zoltan Nagy](https://nagy.caee.utexas.edu/team/prof-zoltan-nagy-phd/), Assistant Professor at The University of Texas at Austin, Department of Civil, Architectural, and Environmental Engineering.
## License
The MIT License (MIT) Copyright (c) 2019, José Ramón Vázquez-Canteli
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
