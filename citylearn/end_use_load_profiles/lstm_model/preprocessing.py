from typing import Any, List, Mapping, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from citylearn.building import Building
from citylearn.data import get_settings
from citylearn.preprocessing import Normalize, PeriodicNormalization

def preprocess_df(config:  Mapping[str, Any], df: pd.DataFrame, train_references: List[int] = None, validation_references: List[int] = None, test_references: List[int] = None) -> Mapping[str, Any]:
    ideal_reference = 0
    # # including in training makes model worse so will exclude for now. 
    # # Nevertheless, prediction for free-float is still decent despite not including it in the training.
    # free_float_reference = 1 
    partial_references = [2, 3, 4, 5]
    train_references = partial_references[:2] if train_references is None else train_references
    validation_references = [partial_references[2]] if validation_references is None else validation_references
    test_references = [partial_references[3]] if test_references is None else test_references
    
    # observation names
    observation_names = get_settings()['schema']['template']['buildings']['Building_1']['dynamics']['attributes']['input_observation_names']
    target = observation_names[-1]
    periodic_observations = Building.get_periodic_observation_metadata()
    
    # periodic normalization
    for k, v in periodic_observations.items():
        result = df[k]*PeriodicNormalization(x_max=v[-1])
        result = pd.DataFrame(result.tolist(), index=result.index)
        df[f'{k}_sin'] = result[0].tolist()
        df[f'{k}_cos'] = result[1].tolist()

    # set min-max normalization limits
    normalization_minimum = df[observation_names].min().values.tolist()
    normalization_maximum = df[observation_names].max().values.tolist()

    # min-max normalization
    for c in observation_names:
        df[c] = df[c]*Normalize(df[c].min(), df[c].max())

    # check to make share there is 1-year worth of data
    months = list(sorted(df['month'].unique()))
    assert len(months) == 12, f'Expected 12 months, got {len(months)} months.'
    
    # training data are ideal load data for every three month step beginning from January
    # and free-float load and 2 partial load datasets for entire year
    train_df = df[
        ((df['reference']==ideal_reference) & (df['month'].isin([months[i] for i in range(0, len(months), 3)])))
        | (df['reference'].isin(train_references))
    ][observation_names].copy()
    X_train, y_train = sliding_windows(train_df.to_numpy(), config['lb'], 1)
    train_df, train_loader = dataset_dataloader(X_train, y_train, config['batch_size'])
    
    # validation data are ideal load data for every three month step beginning from February
    # and free-float load and 1 partial load dataset for entire year
    validation_df = df[
        ((df['reference']==ideal_reference) & (df['month'].isin([months[i] for i in range(1, len(months), 3)])))
        | (df['reference'].isin(validation_references))
    ][observation_names].copy()
    X_val, y_val = sliding_windows(validation_df.to_numpy(), config['lb'], 1)
    val_df, val_loader = dataset_dataloader(X_val, y_val, config['batch_size'])

    # test data are ideal load data for every three month step beginning from March
    # and free-float load and 1 partial load dataset for entire year
    test_df = df[
        ((df['reference']==ideal_reference) & (df['month'].isin([months[i] for i in range(2, len(months), 3)])))
        | (df['reference'].isin(test_references))
    ].copy()
    test_df_by_season = test_df.copy()
    test_df = test_df[observation_names].copy()
    X_test, y_test = sliding_windows(test_df.to_numpy(), config['lb'], 1)
    test_df, test_loader = dataset_dataloader(X_test, y_test, config['batch_size'])
    
    # seasonal test data is same as test data but in 3-month sequences
    test_df_by_season = [
        test_df_by_season[test_df_by_season['month'].isin(months[i:i+3])][observation_names] 
        for i in range(0, len(months), 3)
    ]
    test_loader_by_season = [dataset_dataloader(
        *sliding_windows(df.to_numpy(), config['lb'], 1), 
        config['batch_size']
    )[1] for df in test_df_by_season]
    
    return {
        'temp_limits': {
            'min': normalization_minimum[observation_names.index(target)],
            'max': normalization_maximum[observation_names.index(target)]
        },
        'loaders': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'test_by_season':  test_loader_by_season
        },
        'train': {
            'X': X_train,
            'y': y_train
        },
        'val': {
            'X': X_val,
            'y': y_val
        },
        'test': {
            'X': X_test,
            'y': y_test
        },
        'observation_metadata': {
            'input_observation_names': observation_names,
            'input_normalization_minimum': normalization_minimum,
            'input_normalization_maximum': normalization_maximum, 
        }
    }

def dataset_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = None, drop_last: bool = None) -> Tuple[TensorDataset, DataLoader]:
    shuffle = True if shuffle is None else shuffle
    drop_last = True if drop_last is None else drop_last
    tensor = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(tensor, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    
    return tensor, loader

def sliding_windows(data: np.ndarray, seq_length: int, output_len: int):
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