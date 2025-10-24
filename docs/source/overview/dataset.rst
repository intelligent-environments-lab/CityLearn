.. _dataset-page:

=======
Dataset
=======

CityLearn makes use of datasets that are a collection of data files. The data files are used to define the simulation environment as well as provide some observation values. See :ref:`dataset-data-files-section` for more information.

.. _dataset-data-files-section:

Data Files
**********

The data files refer to flat files containing time series data that are used to set observations that are agent action agnostic (i.e. observations that are not a function of the control actions). These files are referenced in the environment :code:`schema.json` and read when :py:class:`citylearn.citylearn.CityLearnEnv` is initialized. The data files are desrcribes as follows:

Building Data File
==================

The building file is a :code:`csv` file that contains a building's temporal (calendar), end-use loads, occupancy, solar generation and indoor environment variables time series data. There are as many building files as buildings in the environment. The end-use loads, occupancy, solar generation and indoor environment data may come from simulation in energy modeling software e.g., `EnergyPlus <https://energyplus.net>`_ or from smart meter or from a Building Automation System (BAS). The file structure is shown in the snippet below:

.. include:: ../../../citylearn/data/baeda_3dem/Building_1.csv
    :code: text
    :end-line: 6

Weather Data File
=================

The weather file is a :code:`.csv` file that contains outdoor weather variables time series for the desired simulation geographical location. It is used as the source for :py:attr:`citylearn.building.Building.weather`, which is the source for weather related observations. `Typical Meteorological Year <https://energyplus.net/weather>`_ (TMY) or Actual Meteorological Year (AMY) data can be used. The file structure is shown in the snippet below:

.. include:: ../../../citylearn/data/citylearn_challenge_2022_phase_1/weather.csv
    :code: text
    :end-line: 6

Carbon Intensity Data File
==========================

The carbon intensity file is a :code:`.csv` file that contains CO:sub:`2` emission rate time series. It is used as the source for :py:attr:`citylearn.building.Building.carbon_intensity`, which is the source for the `carbon_intensity` observation. The data can be sourced from grid operators e.g. `ERCOT <https://www.ercot.com/gridinfo/generation>`_, `NYISO <http://mis.nyiso.com/public/P-63list.htm>`_ or third-party sources `WattTime <https://www.watttime.org>`_. The file structure is shown in the snippet below:

.. include:: ../../../citylearn/data/citylearn_challenge_2022_phase_1/carbon_intensity.csv
    :code: text
    :end-line: 6

Pricing Data File
=================

The pricing file is a :code:`.csv` file that contains current time-step and forecasted electricity price time series. It is used as the source for :py:attr:`citylearn.building.Building.pricing`, which is the source for pricing related observations. The data can be sourced from specific utility providers for a desired location e.g. `Edison <https://www.sce.com/residential/rates/Time-Of-Use-Residential-Rate-Plans>`_. The file structure is shown in the snippet below:

.. include:: ../../../citylearn/data/citylearn_challenge_2022_phase_1/pricing.csv
    :code: text
    :end-line: 6

LSTM Model File
===============

The LSTM model file is an optional PyTorch state dictionary used to initialize the :code:`cooling_dynamics` and :code:`heating_dynamics` temperature dynamics model attributes in :py:class:`citylearn.building.DynamicsBuilding` and its descendant classes.

Schema Data File
================

The schema file is a :code:`.json` file that references all other data files and is used to define the simulation environment. Refer to :ref:`schema-page` for more information.

.. warning::
   If using version ``2.1.2`` or earlier, do not change the order of columns in any of the :code:`.csv` data files!