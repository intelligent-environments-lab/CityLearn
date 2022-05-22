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

Cite CityLearn
--------------
   .. [1] Vázquez-Canteli, J. R., Dey, S., Henze, G., & Nagy, Z. (2020).
          *CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management.*
          `doi: 10.48550/arXiv.2012.10504 <https://doi.org/10.48550/arXiv.2012.10504>`_
   .. [2] José R. Vázquez-Canteli, Jérôme Kämpf, Gregor Henze, and Zoltan Nagy. (2019).
          *CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep Reinforcement Learning.*
          In Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '19). Association for Computing Machinery, New York, NY, USA, 356–357.
          `doi: 10.1145/3360322.3360998 <https://doi.org/10.1145/3360322.3360998>`_

.. Related Publications
.. --------------------

..    .. [1] Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018).
..           *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)*.
..           Neurocomputing 307 (2018) 72-77,
..           `doi: 10.1016/j.neucom.2018.03.067 <https://doi.org/10.1016/j.neucom.2018.03.067>`_.
..    .. [2] Christ, M., Kempa-Liehr, A.W. and Feindt, M. (2016).
..           *Distributed and parallel time series feature extraction for industrial big data applications*.
..           Asian Conference on Machine Learning (ACML), Workshop on Learning on Big Data (WLBD).
..           `<https://arxiv.org/abs/1610.07717v1>`_.
..    .. [3] Kempa-Liehr, A.W., Oram, J., Wong, A., Finch, M. and Besier, T. (2020).
..           *Feature engineering workflow for activity recognition from synchronized inertial measurement units*.
..           In: Pattern Recognition. ACPR 2019. Ed. by M. Cree et al. Vol. 1180.
..           Communications in Computer and Information Science (CCIS).
..           Singapore: Springer 2020, 223–231.
..           `doi: 10.1007/978-981-15-3651-9_20 <https://doi.org/10.1007/978-981-15-3651-9_20>`_.


.. image:: ../../images/citylearn_diagram.png
   :scale: 30 %
   :alt: demand response
   :align: center

.. toctree::
   usage
   api/modules
   :maxdepth: 2
   :caption: Contents:
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
