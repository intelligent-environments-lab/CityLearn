import os
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def import_data(dir_path = 'simulation\\data\\lstm_train_data\\'):
    """"
    This function is used for import a dataframe from the csv directory
    """

    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    # Set the current working directory
    os.chdir('../')

    # Load all file in dir_path
    file_list = []

    # Initialize the dict of the dataframe
    df_dict = {}

    # Iterate directory to return a file list of all document that directory contains
    for path in os.listdir(dir_path):
        print(path)
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(path)
            df = pd.read_csv(dir_path + path)

            # Setting variables
            df['simulation_reference'] = df['simulation_reference'].astype(str)

            # Fix timeseries to datetime data type
            # df['datetime'] = pd.to_datetime(
                # dict(year=df['year'], month=df['month'], day=df['day'], hour=df['hour'], minute=df['minute']))

            # Sin Cos variable transformation
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

            df['sin_DoW'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['cos_DoW'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            df_dict[df.resstock_building_id.unique()[0]] = df

    print('Dataset imported well!')

    return df_dict


# SLIDING WINDOW
def sliding_windows(data, seq_length, output_len):
    """
    Check that the variable to be predicted is the last column of the dataframe
    :param data: dataframe
    :param seq_length: lookback
    :param output_len: how many timetep ahead will be predicted
    :return: x = matrix [number of timestep - lookback, lookback, number of input variables];
             y = matrix [number of timestep - lookback, number of output variables]
    """
    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[(i+1):(i + 1 + seq_length), :-1]
        T_lag = data[i:(i+seq_length), -1]
        _y = data[(i + seq_length):(i + seq_length + output_len), -1]  # If you want to predict more than one timestamp
        _x = np.column_stack([_x, T_lag])
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)