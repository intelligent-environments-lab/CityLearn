# CityLearn
CityLearn is an open source OpenAI Gym environment for the implementation of reinforcement learning (RL) for simulated demand response scenarios in buildings and cities. Its objective is to facilitiate the design of RL agents to manage energy more efficiently in cities, and also standardize this field of research such that different algorithms can be easily compared with each other.
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/dr.jpg)
## Description
Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Demand response is the coordination of the electricity consuming agents (i.e. buildings) in order to flatten the overall curve of electrical demand.
CityLearn allows the research community to explore the use of reinforcement learning to coordinate the electricity consumption in a district with multiple buildings by controlling the amount of stored energy in the summer period.
## Files
- [main.ipynb](/main.ipynb): Example of the implementation of a reinforcement learning agent ([DDPG](https://arxiv.org/abs/1509.02971)) in a single building in ```CityLearn```
- [citylearn.py](/citylearn.py): Contains the ```CityLearn``` environment and the functions ```building_loader()``` and ```autosize()```
- [energy_models.py](/energy_models.py): Contains the classes ```Building```, ```HeatPump``` and ```EnergyStorage```, which are called by the ```CityLearn``` class
- [agent.py](/agent.py): Implementation of the Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)) RL algorithm. This file must be modified with any other RL implementation, which can then be run in the [main.ipynb](/main.ipynb) file.
- [reward_function.py](/reward_function.py): Contains the reward function that wraps and modifies the rewards obtained from ```CityLearn```. This function can be modified by the user in order to minimize the cost function of ```CityLearn```.
- [example_rbc.ipynb](/example_rbc.ipynb): Example of the implementation of a manually optimized Rule-based controller (RBC) that can be used as a comparison
### Classes
- CityLearn
  - Building
    - HeatPump
    - EnergyStorage
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/agents.jpg)
### Building
The heating and cooling demands of the buildings are obtained from [CitySim](https://www.epfl.ch/labs/leso/transfer/software/citysim/), a building energy simulator for urban scale analysis. Every building is instantiated by defining its associated energy supply and storage devices.
- Methods
  - ```state_space()``` and ```action_space()``` set the state-action space of each building
  - ```set_storage_heating()``` and ```set_storage_cooling()``` set the state of charge of the ```EnergyStorage``` device to the specified value and within the physical constraints of the system. Returns the total electricity consumption of the building at that time-step.
### Heat pump
The efficiency is given by the coefficient of performance (COP), which is calculated as a function of the outdoor temperature ```T_outdoorAir```, the technical efficiency of the heat pump ```eta_tech```, and the target temperatures ```T_target```.
Any amount of cooling demand of the building that isn't satisfied by the ```EnergyStorage``` device is automatically supplied by the ```HeatPump``` directly to the ```Building```. The ```HeatPump``` is more efficient (has a higher COP) if the outdoor air temperature ```s2``` is lower, and less efficient (lower COP) when the outdoor temperature is higher (typically during the day time). On the other hand, the demand for cooling in the building is higher during the daytime and lower at night. ```COP = cooling_energy_generated/electricity_consumed, COP > 1```
- Methods
  - ```get_max_cooling_power()``` and ```get_max_heating_power()``` compute the maximum amount of heating or cooling that the heat pump can provide based on its nominal power of the compressor and its COP. 
  - ```get_electric_consumption_cooling()``` and ```get_electric_consumption_heating()``` return the amount of electricity consumed by the heat pump for a given amount of supplied heating or cooling energy.
### Energy storage
Storage devices allow heat pumps to store energy that can be later released into the building. Typically every building will have its own storage device, but CityLearn also allows defining a single instance of the ```EnergyStorage``` for multiple instances of the class ```Building```, therefore having a group of buildings sharing a same energy storage device.
- Methods
  - ```charge()``` increases (+) or decreases (-) of the amount of energy stored. The input is the amount of energy as a ratio of the total capacity of the storage device (can vary from -1 to 1). Outputs the energy balance of the storage device.
## Environment variables
### States
- ```s1```: outdoor temperature in Celcius degrees. Same for every building for every time step. 
- ```s2```: hour of day (from 1 to 24). Same for every building for every time step. 
- ```s3```: state of the charge (SOC) of the energy storage device. From 0 (no energy stored) to 1 (at full capacity). It is different for every building and depends on the actions taken by each agent.
### Actions
- ```a```: increase (+) or decrease (-) of the amount of cooling energy stored in the energy storage device. Goes from -0.5 to 0.5 (attempts to decrease/increase the cooling energy stored in the storage device by an amount equivalent to 0.5 times its maximum capacity). In order to decrease the energy stored in the device, the energy must be released into the building. Therefore, the state of charge ```s3``` may not decrease by the same amount as the action ```a``` taken if the demand for cooling energy in the building is lower than ```a```.
### Reward
- ```r```: the reward returned by CityLearn is the electricity consumption of every building for a given hour. Then, the function ```reward_function``` converts these rewards to electricity costs. See ```reward_function.py```, which contains a function that wraps the rewards obtained from the environment. The ```reward_function``` can be customized by the user in order to minimize the cost returned by the environment.
### Cost function
```env.cost()``` sqrt(sum(e^2)). Where 'e' is the sum of the  electricity consumption of all the buildings in a given hour, and sum(e^2) is the sum of the squares of 'e' over the whole simulation period. The objetive of the agent(s) must be to minimize this cost. Minimizing the ```env.cost()``` is achieved when the overall curve of electricity demand is flattened and reduced as much as possible.
## Additional functions
- ```building_loader(demand_file, weather_file, buildings)``` receives a dictionary with all the building instances and their respectives IDs, and loads them with the data of heating and cooling loads from the simulations.
- ```auto_size(buildings, t_target_heating, t_target_cooling)``` automatically sizes the heat pumps and the storage devices. It assumes fixed target temperatures of the heat pump for heating and cooling, which combines with weather data to estimate their hourly COP for the simulated period. The ```HeatPump``` is sized such that it will always be able to fully satisfy the heating and cooling demands of the building. This function also sizes the ```EnergyStorage``` devices, setting their capacity as 3 times the maximum hourly cooling demand in the simulated period.
## Multi-agent coordination
### One building
  - The optimal policy consists on storing cooling energy during the night (when the cooling demand of the building is low and the COP of the heat pump is higher), and releasing the stored cooling energy into the building during the day (high demand for cooling and low COP). 
### Multiple buildings
  - If controlled independently of each other and with no coordination, they will all tend to consume more electricity simultaneously during the same hours at night (when the COPs are highest), raising the price for electricity that they all pay at this time and therefore the electricity cost won't be completely minimized.
### Challenges 
1. Implement an independent RL agent for every building (this has already been done in this example) and try to minimize the scores in the minimum number of episodes for multiple buildings running simultaneously. The algorithm should be properly calibrated to maximize its likelyhood of converging to a good policy (the current example does not converge 100% of the times it is run). 
2. Coordinate multiple decentralized RL agents or a single centralized agent to control all the buildings. The agents could share certain information with each other (i.e. ```s3```), while other variables (i.e. ```s1``` and ```s2```) are aleady common for all the agents. The agents could decide which actions to take sequentially and share this information whith other agents so they can decide what actions they will take. Pay especial attention to whether the environment (as seen by every agent) follows the Markov property or not, and how the states should be defined accordingly such that it is as Markovian as possible.
## Publications
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
