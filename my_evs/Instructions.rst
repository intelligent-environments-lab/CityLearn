Base Case
==========

This is the base case that will be created if no specific template is selected. It serves as a foundation for own modeling and to get an overview for the program.

To initialize the base case and create a project folder, no template needs to be specified:

.. code-block:: bash

    $ emobpy create -n <give a name>

.. Hint::
    Before running this example, install and activate a dedicated environment (a conda environment is recommended).
    
The initialisation creates a folder and file structure as follows:

.. code-block:: bash

    ├── my_evs
    │   └── config_files
    │       ├── DepartureDestinationTrip.csv
    │       ├── DistanceDurationTrip.csv
    │       ├── TripsPerDay.csv
    │       ├── rules.yml
    │   ├── Time-series_generation.ipynb
    │   ├── Step1Mobility.py
    │   ├── Step2DrivingConsumption.py
    │   ├── Step3GridAvailability.py
    │   ├── Step4GridDemand.py
    │   ├── Visualize_and_Export.ipynb


This base case consists of four `.py` files that run the modelling, a `.ipynb` to visualise the results and the `config_files` folder that contains mobility data.

+---------------------------------+-----------------------------------------------------------------------------------+
| File name                       |  Description                                                                      |
+=================================+===================================================================================+
|``config_files/``                | Mobility data files that can be changed in this folder.                           |
+---------------------------------+-----------------------------------------------------------------------------------+
|``Step1Mobility.py``             | Uses :meth:`emobpy.Mobility` to create individual mobility time series with       |
|                                 | vehicle location and distance travelled.                                          |
+---------------------------------+-----------------------------------------------------------------------------------+
|``Step2DrivingConsumption.py``   | Uses :meth:`emobpy.Consumption` to assign vehicles and to model their consumption.|
+---------------------------------+-----------------------------------------------------------------------------------+
|``Step3GridAvailability.py``     | Uses :meth:`emobpy.Availability` to create the grid availability time series.     |
+---------------------------------+-----------------------------------------------------------------------------------+
|``Step4GridDemand.py``           | Uses :meth:`emobpy.Charging` to calculate the grid electricity demand time series.|
+---------------------------------+-----------------------------------------------------------------------------------+
|``Visualize_and_export.ipynb``   | Jupyter Notebook File to view the results. See Visualization.                     |
+---------------------------------+-----------------------------------------------------------------------------------+
|``Time-series_generation.ipynb`` | Jupyter Notebook File to create and visualize all four time series (Recomended).  |
+---------------------------------+-----------------------------------------------------------------------------------+


After initialisation, you have two options: Using jupyter notebook or the python interpreter directly.

Method 1: Using Jupyter notebook
---------------------------------


.. code-block:: bash

    $ jupyter notebook

It will open the notebook in your browser. The document contains all instructions.

.. Warning:: Make sure you have installed jupyter in your activated environment. To install it type in the console ``conda install jupyter``





Method 2: Python interpreter
-----------------------------

Run the script in the following order:

.. code-block:: bash

    $ cd <given name>
    $ python Step1Mobility.py
    $ python Step2DrivingConsumption.py
    $ python Step3GridAvailability.py
    $ python Step4GridDemand.py

The results are saved as pickle files. To read them, two methods can be implemented. Using the DataBase class as described in the Visualize_and_Export.ipynb or by opening the pickle file directly. More information can be found in the `pickle documentation <https://docs.python.org/3/library/pickle.html#module-pickle>`_. 

The pickle file can be opened as follows:

.. code-block:: python

    pickle_in = open("data.pickle","rb")
    data = pickle.load(pickle_in)



