from model_generation import folder, reformat_filepath, run
import numpy as np
import os
import pandas as pd

def run_one_model(df_id, df, n_tries=3):
    model_pth, error_metrics = get_model_pth(df_id, df, n_tries)
    return {
        "model.pth": model_pth,
        "normalization_limits": get_model_normalization_limits(df),
        "error_metrics": error_metrics
    }

def get_model_pth(df_id, df, n_tries):
    errors = []
    for i in range(n_tries):
        errors.append(run(df, i))
    best_i = np.argmin([abs(i[0]["absolute error"]) for i in errors])
    os.system(reformat_filepaths(f"cp {folder}/models/{best_i}.pth {folder}/best_models/{df_id}.pth"))
    for i in range(n_tries):
        os.system(reformat_filepaths(f"rm {folder}/models/{i}.pth"))
        if i != best_i:
            os.system(reformat_filepaths(f"rm -r {folder}/results/{i}"))
    os.system(reformat_filepaths(f"mv {folder}/results/{best_i} {folder}/results/{df_id}"))
    return reformat_filepath(f"{folder}/best_models/{df_id}.pth"), errors[best_i]

def get_model_normalization_limits(df):
    return {
        "max": df.max(axis=0),
        "min": df.min(axis=0)
    }

def reformat_filepaths(s):
    return " ".join([reformat_filepath(i) for i in s.split(" ")])
