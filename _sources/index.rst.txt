.. CityLearn documentation master file, created by
   sphinx-quickstart on Sat May 21 23:17:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CityLearn
=========

CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. Its objective is to facilitiate and standardize the evaluation of RL agents such that different algorithms can be easily compared with each other.

.. image:: ../../images/dr.jpg
   :scale: 30 %
   :alt: demand response
   :align: center

Description
-----------

Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. Demand response is the coordination of electricity consuming agents (i.e. buildings) in order to reshape the overall curve of electrical demand.
CityLearn allows the easy implementation of reinforcement learning agents in a multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent. Currently, CityLearn allows controlling the storage of domestic hot water (DHW), and chilled water (for sensible cooling and dehumidification). CityLearn also includes models of air-to-water heat pumps, electric heaters, solar photovoltaic arrays, and the pre-computed energy loads of the buildings, which include space cooling, dehumidification, appliances, DHW, and solar generation.

.. image:: ../../images/citylearn_diagram.png
   :scale: 30 %
   :alt: demand response
   :align: center

.. toctree::
   usage
   :maxdepth: 2
   :caption: Contents:
   
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
