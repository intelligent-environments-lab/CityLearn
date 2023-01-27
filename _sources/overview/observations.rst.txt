============
Observations
============

The observations the CityLearn are grouped into calendar, weather, district and building categories. The observation values in the calendar, weather and district categories are equal across all buildings in the environment while the building category observations are building-specific. The observations may be pre-calculated, pre-simulated or pre-measured and supplied to the environment through flat :file:`.csv` files. Other observations are dependent on the actions taken by agents thus, are calculated during the simulation runtime.

.. csv-table::
   :file: ../../../assets/tables/citylearn_observations.csv
   :header-rows: 1