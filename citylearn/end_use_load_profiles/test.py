from lstm_model.preprocessing import import_data
import pandas as pd
from typing import Any, Mapping
from model_generation_wrapper import run_one_model

def train_lstm(data: Mapping[int, pd.DataFrame], **kwargs) -> Mapping[int, Mapping[str, Any]]:
    """
    TODO: Satvik & Pavani
    1. Install training repo using pip.
    2. Parse training data and custom kwargs for training to some function in the training package
        that trains and finds a best model for the building
    3. Train the building LSTM and return .pth, normalization limits, & error metrics
    """
    if "n_tries" in kwargs:
        d = {
            df_id: run_one_model(df_id, data[df_id], kwargs["n_tries"])
            for df_id in data
        }
    else:
        d = {
            df_id: run_one_model(df_id, data[df_id])
            for df_id in data
        }
    return d

df_dict = import_data("../../train-citylearn-lstm-temperature-dynamics-model/data/")
df_dict = {
    i: df_dict[i]
    for i in list(df_dict.keys())[:1]
}

train_lstm(df_dict, n_tries=1)

