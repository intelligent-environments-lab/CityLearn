=========
CityLearn
=========

CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities :cite:p:`https://doi.org/10.48550/arxiv.2012.10504, 10.1145/3360322.3360998`. A major challenge for RL in demand response is the ability to compare algorithm performance :cite:p:`VAZQUEZCANTELI20191072`. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

.. image:: ../../assets/images/dr.jpg
   :scale: 30 %
   :alt: demand response
   :align: center

Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. CityLearn allows the easy implementation of reinforcement learning agents in a single or multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent.

Applications
============

CityLearn has been utilized in the following projects and publications:

.. csv-table::
   :file: ../../assets/tables/citylearn_applications.csv
   :header-rows: 1

.. toctree::
   :hidden:
   
   installation
   overview/index
   usage
   api/modules
   citylearn_challenge/index
   references
   :maxdepth: 1
   :caption: Contents:

Cite CityLearn
==============

.. code-block:: bibtex

   @misc{https://doi.org/10.48550/arxiv.2012.10504,
      doi = {10.48550/ARXIV.2012.10504},
      url = {https://arxiv.org/abs/2012.10504},
      author = {Vazquez-Canteli, Jose R and Dey, Sourav and Henze, Gregor and Nagy, Zoltan},
      keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.1},
      title = {CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management},
      publisher = {arXiv},
      year = {2020},
      copyright = {Creative Commons Attribution 4.0 International}
   }

Contributing
============

CityLearn is an open-source project that continues to benefit from community-driven updates and suggestion. Please, visit the `CityLearn GitHub repository <https://github.com/intelligent-environments-lab/CityLearn>`_ to join the community and start contributing!
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
