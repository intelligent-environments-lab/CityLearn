from typing import Union
from model_generation import folder, reformat_filepath, run
import numpy as np
import os
import pandas as pd
import random
import torch

def run_one_model(config, df_id: Union[int, str], df: pd.DataFrame, directory: Union[Path, str], seed: int):
    model, error_metrics = get_model(config, df_id, df, seed, directory)
    normalization_limits = get_model_normalization_limits(df)
    return {
        "model": model,
        "attributes": {
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layer"],
            "lookback": config["lb"],
            "observation_names": [""], # not sure what to put here
            "normalization_minimum": normalization_limits["min"],
            "normalization_maximum": normalization_limits["maximum"]
        },
        "error_metrics": error_metrics
    }

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =  True

def get_model(config, df_id, df, seed, folder):
    os.makedirs(f"{folder}/best_models", exist_ok=True)
    set_random_seeds(seed)
    error = run(config, df, seed)
    os.system(reformat_filepaths(f"rm {folder}/models/{seed}.pth"))
    os.system(reformat_filepaths(f"rm -r {folder}/results/{seed}"))
    os.system(reformat_filepaths(f"mv {folder}/results/{seed} {folder}/results/{df_id}"))
    # return reformat_filepath(f"{folder}/best_models/{df_id}.pth"), error
    return torch.load(reformat_filepath(f"{folder}/models/{filename}.pth")), error

def get_model_normalization_limits(df):
    return {
        "max": df.max(axis=0),
        "min": df.min(axis=0)
    }

def reformat_filepaths(s):
    return " ".join([reformat_filepath(i) for i in s.split(" ")])
