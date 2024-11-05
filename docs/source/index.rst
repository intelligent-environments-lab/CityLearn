=========
CityLearn
=========

.. panels::
   :column: col-lg-12 p-2

   This output is listed in the `Digital Public Goods Alliance Registry <https://digitalpublicgoods.net/registry/>`_ and contributes to the following UN `Sustainable Development Goals (SDGs) <https://sdgs.un.org/goals>`_:

   .. image:: ../../assets/images/un_sdg_7.png
      :alt: United Nations Sustainable Development Goal 7: Affordable and Clean Energy
      :scale: 8 %
      :target: https://sdgs.un.org/goals/goal7

   .. image:: ../../assets/images/un_sdg_11.png
      :alt: United Nations Sustainable Development Goal 11: Sustainable Cities and Communities
      :scale: 8 %
      :target: https://sdgs.un.org/goals/goal11
      
   .. image:: ../../assets/images/un_sdg_13.png
      :alt: United Nations Sustainable Development Goal 13: Climate Action
      :scale: 8 %
      :target: https://sdgs.un.org/goals/goal13

CityLearn is an open source Farama Foundation Gymnasium environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities :cite:p:`https://doi.org/10.48550/arxiv.2012.10504, 10.1145/3360322.3360998, doi:10.1080/19401493.2024.2418813`. A major challenge for RL in demand response is the ability to compare algorithm performance :cite:p:`VAZQUEZCANTELI20191072`. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

.. image:: ../../assets/images/dr.jpg
   :scale: 30 %
   :alt: demand response
   :align: center

Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. CityLearn allows the easy implementation of reinforcement learning agents in a single or multi-agent setting to reshape their aggregated curve of electrical demand by controlling active energy storage for load shifting and heat pump or electric heater power for load shedding.

Applications
************

CityLearn has been utilized in the following projects and publications:

.. csv-table::
   :file: ../../assets/tables/citylearn_applications.csv
   :header-rows: 1

.. toctree::
   :hidden:
   
   installation
   quickstart
   overview/index
   usage/index
   cli
   api/modules
   citylearn_challenge/index
   contributing
   references

Cite CityLearn
**************

.. code-block:: bibtex

   @article{doi:10.1080/19401493.2024.2418813,
      author = {Nweye, Kingsley and Kaspar, Kathryn and Buscemi, Giacomo and Fonseca, Tiago and Pinto, Giuseppe and Ghose, Dipanjan and Duddukuru, Satvik and Pratapa, Pavani and Li, Han and Mohammadi, Javad and Lino Ferreira, Luis and Hong, Tianzhen and Ouf, Mohamed and Capozzoli, Alfonso and Nagy, Zoltan},
      title = {CityLearn v2: energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities},
      journal = {Journal of Building Performance Simulation},
      volume = {0},
      number = {0},
      pages = {1--22},
      year = {2024},
      publisher = {Taylor \& Francis},
      doi = {10.1080/19401493.2024.2418813},
      url = {https://doi.org/10.1080/19401493.2024.2418813},
   }


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`