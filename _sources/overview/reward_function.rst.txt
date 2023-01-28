===============
Reward Function
===============

A reward is calculated and returned each time :py:meth:`citylearn.citylearn.CityLearnEnv.step` is called. The reward time series is also accessible through the :py:attr:`citylearn.citylearn.CityLearnEnv.rewards` property.

CityLearn provides custom reward functions for centralized agent and/or decentralized control architectures:

.. csv-table::
   :file: ../../../assets/tables/citylearn_reward_functions.csv
   :header-rows: 1

Where :math:`e` is a building's net electricity consumption while :math:`E` is the district's net electricity consumption. For rewards that work with bothe centralized and decentralized agents, :math:`e` is interchangeable with :math:`E` in their equation depending on the value of `citylearn.citylearn.CityLearnEnv.central_agent`.

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

