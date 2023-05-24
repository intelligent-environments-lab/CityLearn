===============
Reward Function
===============

A reward is calculated and returned each time :py:meth:`citylearn.citylearn.CityLearnEnv.step` is called. The reward time series is also accessible through the :py:attr:`citylearn.citylearn.CityLearnEnv.rewards` property.

CityLearn provides custom reward functions:

.. csv-table::
   :file: ../../../assets/tables/citylearn_reward_functions.csv
   :header-rows: 1

Where :math:`e` is a building's net electricity consumption, :math:`T_{in}` is a building's indoor dry-bulb temperature, :math:`T_{spt}` is a building's indoor dry-bulb temperature setpoint, :math:`T_{b}` is a building's indoor dry-bulb temperature setpoint comfort band while :math:`E` is the district's net electricity consumption. These rewards are defined for a decentralized single building application and for a centralized agent controlling all buildings, the reward will be the sum of the decentralized values.

How to Point to the Reward Function
***********************************

The reward function to use in a simulation is defined in the :code:`reward_function` key-value of the schema:

.. code:: json

   {
      ...,
      "reward_function": {
         "type": "citylearn.reward_function.RewardFunction",
         ...
      },
      ...
   }

How to Define a Custom Reward Function
**************************************

CityLearn also allows for custom reward functions by inheriting the base :py:class:`citylearn.reward_function.RewardFunction`:

.. include:: ../../../examples/custom_reward_function.py
    :code: python
    :start-line: 11

The schema must then be updated to reference the custom reward function:

.. code:: json

   {
      ...,
      "reward_function": {
         "type": "custom_module.CustomReward",
         ...
      },
      ...
   }