==========================
How to Load an Environment
==========================

Load an Environment Using Schema Filepath
*****************************************

The :ref:`schema-page` filepath can be use to initialize an environment:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    
    schema_filepath = 'path/to/schema.json'
    env = CityLearnEnv(schema_filepath)

This approach is best if using a custom :ref:`dataset-page`.

Load an Environment Using Schema Dictionary Object
**************************************************

Alternatively, the schema can be supplied as a :py:obj:`dict` object. This approach can be used to edit the schema parameter values before constructing the environment. With this approach, the :code:`root_directory` key-value must be explicitly set: See example below:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    from citylearn.utilities import read_json
    
    schema_filepath = 'path/to/schema.json'
    schema = read_json(schema_filepath)
    schema['root_directory'] = 'path/to'
    env = CityLearnEnv(schema)

Load an Environment Using Named Dataset
***************************************

CityLearn provides some data files that are contained in named datasets including those that have been used in :ref:`The CityLearn Challenge <citylearn-challenge-page>`. These datasets names can be used in place of schema filepaths or :code:`dict` objects to initialize an environment. To get the dataset names run:

.. code:: python

    from citylearn.data import DataSet

    dataset_names = DataSet.get_names()
    print(dataset_names)

    # output
    # ['citylearn_challenge_2020_climate_zone_1', 'citylearn_challenge_2020_climate_zone_2', ...]

Initialize the environment using any of the valid names:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    
    env = CityLearnEnv('citylearn_challenge_2020_climate_zone_1')

The dataset can also be download to a path of choice for inspection. The following code copies a dataset to the current directory:

.. code:: python

    from citylearn.data import DataSet

    DataSet.copy('citylearn_challenge_2020_climate_zone_1')