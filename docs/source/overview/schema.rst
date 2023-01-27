======
Schema
======

The schema is a :file:`.json` file containing key-value pairs that define the parameters to use in constructing a :py:class:`citylearn.citylearn.CityLearnEnv` object (environment). The aim of the schema is to provide an interface that is analogous to the :file:`.idf` used to define an `EnergyPlus <https://energyplus.net>`_ model.

How to Load an Environment using the Schema
===========================================

To construct an environment, the schema can be supplied as a filepath:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    
    schema_filepath = 'path/to/schema.json'
    env = CityLearnEnv(schema_filepath)

Alternatively, the schema can be supplied as loaded :file:`.json` that has been converted to a :py:obj:`dict`. This approach can be used to edit the parameter values before constructing the environment. Using this approach, the :code:`root_directory` key-value must be explicitly set: See example below:  

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    from citylearn.utilities import read_json
    
    schema_filepath = 'path/to/schema.json'
    schema = read_json(schema_filepath)
    schema['root_directory'] = 'path/to'
    env = CityLearnEnv(schema)

Schema Definition
=================

The key-value pairs in the schema are summarized in the table:

.. csv-table::
   :file: ../../../assets/tables/citylearn_schema_definition.csv
   :header-rows: 1

An Example Schema
=================

An example schema is shown below:

.. include:: ../../../citylearn/data/citylearn_challenge_2021/schema.json
    :code: json