.. _installation-page:

============
Installation
============

Install the latest CityLearn version from PyPi with the :code:`pip` command:

.. code-block:: console

   pip install CityLearn

Alternatively, install a specific version:

.. code-block:: console

   pip install CityLearn==2.0.0

.. warning::
   Version ``1.8.0`` and earlier do not support temperature dynamic buildings class :py:class:`citylearn.building.DynamicsBuilding` and its descendant classes. Thus, `cooling_device` and `heating_device` control for partial load is not supported by these versions.

.. warning::
   Version ``1.1.0`` and earlier are not installable via :code:`pip`. Instead, clone earlier versions like:

   .. code-block:: bash

        git clone -b v1.1.0 https://github.com/intelligent-environments-lab/CityLearn.git