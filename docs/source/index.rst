.. CityLearn documentation master file, created by
   sphinx-quickstart on Sat May 21 23:17:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CityLearn
=========

.. toctree::
   usage
   environment
   citylearn_challenge/years
   api/modules
   :maxdepth: 1
   :caption: Contents:

Description
-----------

CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. Its objective is to facilitate and standardize the evaluation of RL agents such that different algorithms can be easily compared with each other.

.. image:: ../../assets/images/dr.jpg
   :scale: 30 %
   :alt: demand response
   :align: center

Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. Demand response is the coordination of electricity consuming agents (i.e. buildings) in order to reshape the overall curve of electrical demand.
CityLearn allows the easy implementation of reinforcement learning agents in a multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent.

Cite CityLearn
--------------
   .. [1] Vázquez-Canteli, J. R., Dey, S., Henze, G., & Nagy, Z. (2020).
          *CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management.*
          `doi: 10.48550/arXiv.2012.10504 <https://doi.org/10.48550/arXiv.2012.10504>`_
   .. [2] José R. Vázquez-Canteli, Jérôme Kämpf, Gregor Henze, and Zoltan Nagy. (2019).
          *CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep Reinforcement Learning.*
          In Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '19). Association for Computing Machinery, New York, NY, USA, 356–357.
          `doi: 10.1145/3360322.3360998 <https://doi.org/10.1145/3360322.3360998>`_

Related Publications
--------------------
   .. [3] Vazquez-Canteli, J. R., Henze, G., & Nagy, Z. (2020, November).
          *MARLISA: Multi-agent reinforcement learning with iterative sequential action selection for load shaping of grid-interactive connected buildings.*
          In Proceedings of the 7th ACM international conference on systems for energy-efficient buildings, cities, and transportation (pp. 170-179).
          `doi: 10.1145/3408308.3427604 <http://dx.doi.org/10.1145/3408308.3427604>`_

   .. [4] Vázquez-Canteli, J. R., & Nagy, Z. (2019). 
          *Reinforcement learning for demand response: A review of algorithms and modeling techniques.*
          Applied energy, 235, 1072-1089.
          `doi: 10.1016/j.apenergy.2018.11.002 <https://doi.org/10.1016/j.apenergy.2018.11.002>`_
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
