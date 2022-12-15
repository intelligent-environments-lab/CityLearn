CityLearn Environment
=====================

Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. Demand response is the coordination of electricity consuming agents (i.e. buildings) in order to reshape the overall curve of electrical demand. CityLearn is an OpenAI Gym Environment that allows the easy implementation of reinforcement learning agents in a single or multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent. Below are details of the CityLearn `environment <#environment>`__,
`observations <#observations>`__, `actions <#actions>`__ and
`rewards <#rewards>`__.

Environment
-----------

CityLearn is an OpenAI Gym environment for the easy implementation of RL agents in a demand response setting to reshape the aggregated curve of electricity demand by controlling the energy storage of a diverse set of electrified buildings in a district. Its main objective is to facilitate and standardize the evaluation of RL agents, such that it enables benchmarking of different algorithms.

.. image:: ../../assets/images/citylearn_diagram.png
   :scale: 30 %
   :alt: demand response
   :align: center

CityLearn includes energy models of buildings and electric devices including air-to-water heat pumps, electric heaters and batteries. A collection of buildings energy models make up a virtual district. In each building, an air-to-water heat pump may be used to meet the space cooling, space heating and domestic hot water demand. Alternatively, space heating and domestic hot water demand can be satisfied through electric heaters. Buildings may also possess a combination of water tanks and batteries to store energy for space cooling, space heating, domestic hot water and non-shiftable (plug) loads. These storage devices are charged by the same electric device that satisfies the end-use that the stored energy is for. All electric devices as well as plug loads consume electricity from the main grid.

Photovoltaic (PV) arrays may be included in the buildings to offset all or part of the electricity consumption from the grid by allowing the buildings to generate their own electricity.

The RL agents control the storage of chilled water, hot water and electricity by deciding how much cooling, heating and electrical energy to store or release at any given time. CityLearn guarantees that, at any time, the heating, cooling, domestic hot water and non-shiftable energy demand of the building are satisfied regardless of the actions of the controller by utilizing pre-computed or pre-measured demand of the buildings. An internal backup controller guarantees that the electric devices prioritize satisfying the energy demand of the building before storing energy in the storage devices.

Observations
------------

The observations in the CityLearn environment are grouped into calendar, weather, district and building categories. The initial three categories of observations are common to all agents in the environment while the last is building specific. The observations may be pre-simulated/pre-measured and supplied to the environment through flat ``.csv`` files or are calculated during the simulation runtime.

+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Category | Name                                       | Source                 | Unit            | Description                                                                                                                                                                                                                                                     |
+==========+============================================+========================+=================+=================================================================================================================================================================================================================================================================+
| Calendar | month                                      | `building_id.csv`      | -               | Month of year ranging from 1 (January) through 12 (December).                                                                                                                                                                                                   |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Calendar | day_type                                   | `building_id.csv`      | -               | Day of week ranging from 1 (Monday) through 7 (Sunday).                                                                                                                                                                                                         |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Calendar | hour                                       | `building_id.csv`      | -               | Hour of day ranging from 1 to 24.                                                                                                                                                                                                                               |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Calendar | daylight_savings_status                    | `building_id.csv`      | -               | Boolean that indicates if the current day is daylight savings period. 0 indicates that the buildings have not changed its electricity consumption profiles due to daylight savings, while 1 indicates the period in which the buildings may have been affected. |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_dry_bulb_temperature               | `weather.csv`          | C               | Outdoor dry bulb temperature.                                                                                                                                                                                                                                   |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_dry_bulb_temperature_predicted_6h  | `weather.csv`          | C               | Outdoor dry bulb temperature predicted 6 hours ahead.                                                                                                                                                                                                           |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_dry_bulb_temperature_predicted_12h | `weather.csv`          | C               | Outdoor dry bulb temperature predicted 12 hours ahead.                                                                                                                                                                                                          |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_dry_bulb_temperature_predicted_24h | `weather.csv`          | C               | Outdoor dry bulb temperature predicted 24 hours ahead                                                                                                                                                                                                           |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_relative_humidity                  | `weather.csv`          | %               | Outdoor relative humidity.                                                                                                                                                                                                                                      |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_relative_humidity_predicted_6h     | `weather.csv`          | %               | Outdoor relative humidity predicted 6 hours ahead.                                                                                                                                                                                                              |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_relative_humidity_predicted_12h    | `weather.csv`          | %               | Outdoor dry bulb temperature predicted 12 hours ahead.                                                                                                                                                                                                          |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | outdoor_relative_humidity_predicted_24h    | `weather.csv`          | %               | Outdoor dry bulb temperature predicted 24 hours ahead.                                                                                                                                                                                                          |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | diffuse_solar_irradiance                   | `weather.csv`          | W/m2            | Diffuse solar irradiance.                                                                                                                                                                                                                                       |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | diffuse_solar_irradiance_predicted_6h      | `weather.csv`          | W/m2            | Diffuse solar irradiance predicted 6 hours ahead.                                                                                                                                                                                                               |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | diffuse_solar_irradiance_predicted_12h     | `weather.csv`          | W/m2            | Diffuse solar irradiance predicted 12 hours ahead.                                                                                                                                                                                                              |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | diffuse_solar_irradiance_predicted_24h     | `weather.csv`          | W/m2            | Diffuse solar irradiance predicted 24 hours ahead.                                                                                                                                                                                                              |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | direct_solar_irradiance                    | `weather.csv`          | W/m2            | Direct solar irradiance.                                                                                                                                                                                                                                        |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | direct_solar_irradiance_predicted_6h       | `weather.csv`          | W/m2            | Direct solar irradiance predicted 6 hours ahead.                                                                                                                                                                                                                |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | direct_solar_irradiance_predicted_12h      | `weather.csv`          | W/m2            | Direct solar irradiance predicted 12 hours ahead.                                                                                                                                                                                                               |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Weather  | direct_solar_irradiance_predicted_24h      | `weather.csv`          | W/m2            | Direct solar irradiance predicted 24 hours ahead.                                                                                                                                                                                                               |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| District | carbon_intensity                           | `carbon_intensity.csv` | kgCO2/kWh       | Grid carbon emission rate.                                                                                                                                                                                                                                      |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | indoor_dry_bulb_temperature                | `building_id.csv`      | C               | Zone volume-weighted average building dry bulb temperature.                                                                                                                                                                                                     |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | average_unmet_cooling_setpoint_difference  | `building_id.csv`      | C               | Zone volume-weighted average difference between `indoor_dry_bulb_temperature` and cooling temperature setpoints.                                                                                                                                                |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | indoor_relative_humidity                   | `building_id.csv`      | %               | Zone volume-weighted average building relative humidity.                                                                                                                                                                                                        |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | non_shiftable_load                         | `building_id.csv`      | kWh             | Total building non-shiftable plug and equipment loads.                                                                                                                                                                                                          |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | solar_generation                           | `building_id.csv`      | kWh             | PV electricity generation.                                                                                                                                                                                                                                      |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | cooling_storage_soc                        | Runtime calculation    | kWh/kWhcapacity | State of the charge (SOC) of the `cooling_storage` from 0 (no energy stored) to 1 (at full capacity).                                                                                                                                                           |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | heating_storage_soc                        | Runtime calculation    | kWh/kWhcapacity | State of the charge (SOC) of the `heating_storage` from 0 (no energy stored) to 1 (at full capacity).                                                                                                                                                           |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | dhw_storage_soc                            | Runtime calculation    | kWh/kWhcapacity | State of the charge (SOC) of the `dhw_storage` (domestic hot water storage) from 0 (no energy stored) to 1 (at full capacity).                                                                                                                                  |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | electrical_storage_soc                     | Runtime calculation    | kWh/kWhcapacity | State of the charge (SOC) of the `electrical_storage` from 0 (no energy stored) to 1 (at full capacity).                                                                                                                                                        |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | net_electricity_consumption                | Runtime calculation    | kWh             | Total building electricity consumption.                                                                                                                                                                                                                         |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | electricity_pricing                        | `pricing.csv`          | $/kWh           | Electricity rate.                                                                                                                                                                                                                                               |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | electricity_pricing_predicted_6h           | `pricing.csv`          | $/kWh           | Electricity rate predicted 6 hours ahead.                                                                                                                                                                                                                       |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | electricity_pricing_predicted_12h          | `pricing.csv`          | $/kWh           | Electricity rate predicted 12 hours ahead.                                                                                                                                                                                                                      |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Building | electricity_pricing_predicted_24h          | `pricing.csv`          | $/kWh           | Electricity rate predicted 24 hours ahead.                                                                                                                                                                                                                      |
+----------+--------------------------------------------+------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

These observations can be specified to be active/inactive during simulation in ``schema.json``.

Actions
-------

The actions specify the amount of energy by which the available storage devices in a building (multi-agent) or district of buildings (central agent) are charged/discharged. In a multi-agent setup, 1 agent controls all storage devices in 1 building i.e. provides as many actions as storage devices in 1 building whereas in a central agent setup, 1 agent controls all storage devices in all buildings i.e. provides as many actions as storage devices in the entire district.

================== ====================== ===============
Name               Controlled Storage     Unit
================== ====================== ===============
cooling_storage    ``cooling_storage``    kWh/kWhcapacity
heating_storage    ``heating_storage``    kWh/kWhcapacity
dhw_storage        ``dhw_storage``        kWh/kWhcapacity
electrical_storage ``electrical_storage`` kWh/kWhcapacity
================== ====================== ===============

**NOTE:** The CityLearn Challenge 2022 only utilizes ``electrical_storage``.

Rewards
-------

CityLearn provides custom reward functions for multi-agent and central agent setups that are described in the `docs <https://intelligent-environments-lab.github.io/CityLearn/api/citylearn.reward_function.html>`__. The path to the function for use in simulation is defined in ``schema.json`` e.g.:

.. code:: json

   "reward_function": {"type": "citylearn.reward_function.RewardFunction"}

CityLearn also allows for custom reward functions by inheriting the `citylearn.reward_function.RewardFunction <https://intelligent-environments-lab.github.io/CityLearn/api/citylearn.reward_function.html#citylearn.reward_function.RewardFunction>`__ class. An example is:

.. code:: python

   from typing import List
   from citylearn.reward_function import RewardFunction

   class CustomReward(RewardFunction):
       def __init__(self, agent_count: int, electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float]):
           super().__init__(agent_count, electricity_consumption=electricity_consumption, carbon_emission=carbon_emission, electricity_price=electricity_price)
           
       def calculate(self) -> List[float]:
           """Calculates custom user-defined multi-agent reward.
           
           Reward is the `carbon_emission` for each building.
           """

           return list(self.carbon_emission)

``schema.json`` must then be updated to reference the custom reward
e.g.:

.. code:: json

   "reward_function": {"type": "custom_module.CustomReward"}

Each time the `step function <https://intelligent-environments-lab.github.io/CityLearn/api/citylearn.citylearn.html#citylearn.citylearn.CityLearnEnv.step>`__ is called during simulation, the ``electricity_consumption``, ``carbon_emission`` and ``electricity_price`` properties of the reward class are updated and the custom reward is calculated.

Cost Functions
--------------

CityLearn provides a set of cost metrics that quantify the simulated district’s energy flexibility and performance while under the control of the agent(s). These metrics include the following and their detailed definitions are provided in the `docs <https://intelligent-environments-lab.github.io/CityLearn/api/citylearn.cost_function.html#module-citylearn.cost_function>`__:

1. ``average_daily_peak`` - Rolling mean of daily net electricity consumption peaks.
2. ``carbon_emissions`` - Rolling sum of carbon emissions.
3. ``load_factor`` - Difference between 1 and the rolling mean ratio of rolling mean demand to rolling peak demand over a specified period.
4. ``net_electricity_consumption`` - Rolling sum of net electricity consumption.
5. ``peak_demand`` - Net electricity consumption peaks.
6. ``price`` - Rolling sum of electricity price.
7. ``quadratic`` - Rolling sum of net electricity consumption raised to the power of 2.
8. ``ramping`` - Rolling sum of absolute difference in net electric consumption between consecutive time steps.

**NOTE:** The CityLearn Challenge 2022 only utilizes ``carbon_emissions`` and ``price`` cost functions.
