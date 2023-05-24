.. _schema-page:

======
Schema
======

The schema is a :file:`.json` file containing key-value pairs that define the parameters to use in constructing a :py:class:`citylearn.citylearn.CityLearnEnv` object (environment). The aim of the schema is to provide an interface that is analogous to the :file:`.idf` used to define an `EnergyPlus <https://energyplus.net>`_ model.

Schema Definition
*****************

The key-value pairs in the schema are summarized in the table:

.. csv-table::
   :file: ../../../assets/tables/citylearn_schema_definition.csv
   :header-rows: 1

An Example Schema
*****************

An example schema is shown below:

.. include:: ../../../citylearn/data/baeda_3dem/schema.json
    :code: json