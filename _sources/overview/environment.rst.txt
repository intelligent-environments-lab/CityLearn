===========
Environment
===========

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters. 

.. image:: ../../../assets/images/citylearn_systems.png
   :alt: demand response
   :align: center

Buildings may have a combination of thermal storage tanks and batteries for active energy storage that provide continuous load shifting energy flexibility services. These storage devices may be used at peak or expensive periods to meet space cooling, space heating, domestic hot water and non-shiftable (plug) loads. The storage devices are charged by the cooling or heating devices (heat pump or electric heater) that satisfies the end-use the stored energy is for. All electric devices as well as plug loads consume electricity from the main grid. Photovoltaic (PV) arrays may be included in the buildings to offset all or part of the electricity consumption from the grid by allowing the buildings to generate their own electricity.

The building devices can be autosized to meet the building's needs (see :py:meth:`citylearn.building.Building.autosize_cooling_device`, :py:meth:`citylearn.building.Building.autosize_heating_device`, :py:meth:`citylearn.building.Building.autosize_dhw_device`, :py:meth:`citylearn.building.Building.autosize_cooling_storage`, :py:meth:`citylearn.building.Building.autosize_heating_storage`, :py:meth:`citylearn.building.Building.autosize_dhw_storage`, :py:meth:`citylearn.building.Building.autosize_electrical_storage`, :py:meth:`citylearn.building.Building.autosize_pv`).

An internal backup controller guarantees that the cooling and heating devices prioritize satisfying the building loads before storing energy in the storage devices. The backup controller also guarantees that the storage devices do not discharge more energy than is needed to meet the unsatisfied building loads.

The legacy CityLearn guarantees that, at any time, the ideal space cooling, space heating, domestic hot water and non-shiftable building loads are satisfied by utilizing pre-computed or pre-measured demand of the buildings. Thus, ideal indoor dry-bulb temperature is always maintained and load shedding is not possible.

Since CityLearn version :code:`2.0.0`, building indoor dry-bulb temperature can be dynamic to provide load shedding energy flexibility services compared to earlier versions where the temperature was pre-computed/pre=measured and remained static during simulation. This is achieved through an LSTM model that encodes a building's temperature evolution as a result of supplied cooling or heating energy :cite:p:`PINTO2021117642`. Thus, partial load satisfaction is made possible by controlling the cooling and heating device power which then influences temperature change.

Since CityLearn version :code:`2.1.0`, power outages can be simulated where buildings can only make of their available distributed energy resources including storage devices and PV system to satisfy end-use loads otherwise, risk thermal discomfort and unserved energy during the outage period. During normal operation i.e., when there is no power outage, there is unlimited supply from the grid.

RBC, RL or MPC agent(s) control the active storage devices by determining how much energy to store or release, and the cooling and heating device by determining their supply power at each control time step.