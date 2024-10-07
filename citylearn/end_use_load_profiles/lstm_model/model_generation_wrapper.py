import random
from typing import Any, Mapping, Tuple
import numpy as np
import pandas as pd
import torch
from citylearn.end_use_load_profiles.lstm_model.model import LSTM
from citylearn.end_use_load_profiles.lstm_model.model_generation import run

def run_one_model(config: Mapping[str, Any], df: pd.DataFrame, seed: int) -> Mapping[str, Any]:
    model, observation_metadata, error_metrics = get_model(config, df, seed)
    
    return {
        'model': model,
        'attributes': {
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layer'],
            'lookback': config['lb'],
            'input_observation_names': observation_metadata['input_observation_names'],
            'input_normalization_minimum': observation_metadata['input_normalization_minimum'],
            'input_normalization_maximum': observation_metadata['input_normalization_maximum']
        },
        'error_metrics': error_metrics
    }

def get_model(config: Mapping[str, Any], df: pd.DataFrame, seed) -> Tuple[LSTM, Mapping[str, Any], Mapping[str, float]]:
    set_random_seeds(seed)
    lstm, observation_metadata, error = run(config, df)
    
    return lstm, observation_metadata, error

def set_random_seeds(seed: int, benchmark: bool = None, deterministic: bool = None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False if benchmark is None else benchmark
    torch.backends.cudnn.deterministic = True if deterministic is None else deterministic