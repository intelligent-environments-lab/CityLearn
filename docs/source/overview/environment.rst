===========
Environment
===========

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of buildings energy models make up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters. 

.. image:: ../../../assets/images/citylearn_diagram.png
   :scale: 30 %
   :alt: demand response
   :align: center

Buildings may have a combination of thermal storage tanks and batteries to store energy that may be used at peak or expensive periods to meet space cooling, space heating, domestic hot water and non-shiftable (plug) loads. These storage devices are charged by the electric device (heat pump or electric heater) that satisfies the end-use the stored energy is for. All electric devices as well as plug loads consume electricity from the main grid. Photovoltaic (PV) arrays may be included in the buildings to offset all or part of the electricity consumption from the grid by allowing the buildings to generate their own electricity.

The building devices can be autosized to meet the building's needs (see :py:meth:`citylearn.building.Building.autosize_cooling_device`, :py:meth:`citylearn.building.Building.autosize_heating_device`, :py:meth:`citylearn.building.Building.autosize_dhw_device`, :py:meth:`citylearn.building.Building.autosize_cooling_storage`, :py:meth:`citylearn.building.Building.autosize_heating_storage`, :py:meth:`citylearn.building.Building.autosize_dhw_storage`, :py:meth:`citylearn.building.Building.autosize_electrical_storage`, :py:meth:`citylearn.building.Building.autosize_pv`).

The RBC, RL or MPC agent(s) control the thermal storage tanks and batteries by determining how much energy to store or release at any given time. CityLearn guarantees that, at any time, the space cooling, space heating, domestic hot water and non-shiftable building loads are satisfied regardless of the actions of the controller by utilizing pre-computed or pre-measured demand of the buildings. An internal backup controller guarantees that the electric devices prioritize satisfying the building loads before storing energy in the storage devices. The backup controller also guarantees that the storage devices do not discharge more energy than is needed to meet the unsatisfied building loads.