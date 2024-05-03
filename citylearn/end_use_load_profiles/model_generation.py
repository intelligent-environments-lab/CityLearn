from lstm_model.classes import LSTM
from lstm_model.preprocessing import import_data, sliding_windows
from lstm_model.training import training
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from time import sleep
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import yaml

def reformat_filepath(filepath):
    return os.path.join(*("/".split(filepath)))

folder = reformat_filepath("citylearn/end_use_load_profiles")

config = yaml.safe_load(reformat_filepath("lstm_model/config.yaml"))

def dataset_dataloader(x, y, BATCH_SIZE, shuffle=True):
    TENSOR = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    LOADER = DataLoader(TENSOR, shuffle=shuffle, batch_size=BATCH_SIZE, drop_last=True)
    return TENSOR, LOADER

def process_df(df, months, train_references=["3", "4", "5"], validation_references=["3", "6"], test_references=["3", "7"]):
    minT, maxT = df["average_indoor_air_temperature"].min(), df["average_indoor_air_temperature"].max()
    df.loc[:, "average_indoor_air_temperature"] -= minT
    df.loc[:, "average_indoor_air_temperature"] /= (maxT - minT)
    train_df = pd.concat([df.loc[df["simulation_reference"].isin(["2"])] for i in range(6)])
    train_df = train_df.loc[train_df["month"].isin([months[i] for i in range(0, len(months), 3)])]
    train_df = pd.concat([train_df, df.loc[df["simulation_reference"].isin(train_references)]])
    validation_df = pd.concat([df.loc[df["simulation_reference"].isin(["2"])] for i in range(6)])
    validation_df = validation_df.loc[validation_df["month"].isin([months[i] for i in range(1, len(months), 3)])]
    validation_df = pd.concat([validation_df, df.loc[df["simulation_reference"].isin(validation_references)]])
    test_df = pd.concat([df.loc[df["simulation_reference"].isin(["2"])] for i in range(6)])
    test_df = test_df.loc[test_df["month"].isin([months[i] for i in range(2, len(months), 3)])]
    test_df = pd.concat([test_df, df.loc[df["simulation_reference"].isin(test_references)]])
    test_df_by_season = [test_df.loc[test_df["month"].isin(months[i:i+3])] for i in range(0, len(months), 3)]
    features = ["sin_month", "cos_month", "sin_DoW", "cos_DoW", "sin_hour", "cos_hour", "cooling_load", "heating_load", "direct_solar_radiation", "diffuse_solar_radiation", "outdoor_air_temperature", "occupant_count"]
    target = "average_indoor_air_temperature"
    train_df = train_df[features + [target]]
    validation_df = validation_df[features + [target]]
    test_df = test_df[features + [target]]
    test_df_by_season = [df[features + [target]] for df in test_df_by_season]
    X_train, y_train = sliding_windows(train_df.to_numpy(), config["lb"], 1)
    train_df, train_loader = dataset_dataloader(X_train, y_train, config["batch_size"])
    X_val, y_val = sliding_windows(validation_df.to_numpy(), config["lb"], 1)
    val_df, val_loader = dataset_dataloader(X_val, y_val, config["batch_size"])
    X_test, y_test = sliding_windows(test_df.to_numpy(), config["lb"], 1)
    test_df, test_loader = dataset_dataloader(X_test, y_test, config["batch_size"])
    test_loader_by_season = [dataset_dataloader(*sliding_windows(df.to_numpy(), config["lb"], 1), config["batch_size"])[1] for df in test_df_by_season]
    return {
        "temp_limits": {
            "min": minT,
            "max": maxT
        },
        "loaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "test_by_season":  test_loader_by_season
        },
        "train": {
            "X": X_train,
            "y": y_train
        },
        "val": {
            "X": X_val,
            "y": y_val
        },
        "test": {
            "X": X_test,
            "y": y_test
        }
    }

def train(lstm, train_loader, val_loader, optimizer, criterion, config, maxT, minT, filename):
    lstm.train()
    data = []
    train_losses, val_losses = [], []
    for epoch in range(config["epochs"]):
        LOSS_TRAIN, LOSS_VAL, \
        ylab_train, ypred_train, \
        ylab_val, ypred_val = training(model=lstm, train_loader=train_loader, val_loader=val_loader,
                                        optimizer=optimizer, criterion=criterion, config=config,
                                        maxT=maxT, minT=minT)
        data.append([np.asarray(ylab_val), np.asarray(ypred_val)])
        train_losses.append(LOSS_TRAIN)
        val_losses.append(LOSS_VAL)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.yscale("log")
    plt.title("Loss by training step")
    plt.savefig(reformat_filepath(f"{folder}/results/{filename}/losses.png"))
    plt.close()
    torch.save(lstm.state_dict(), reformat_filepath(f"{folder}/models/{filename}.pth"))

def eval(model, test_loader, optimizer, maxT, minT, filename):
    model.eval()
    h = model.init_hidden(config["batch_size"], config["device"])
    ypred = []
    ylab = []
    criteria = [
        torch.nn.L1Loss(), # MAE
        torch.nn.MSELoss(), # RMSE
    ]
    losses = []
    cumulative_error_profiles = []
    for batch in test_loader:
        input_test, target_test = batch
        input_test = input_test.to(config["device"])
        target_test = target_test.to(config["device"])
        h = tuple([each.data for each in h])
        output_test, h = model(input_test.float(), h)
        optimizer.zero_grad()
        losses.append([criterion(output_test, target_test.float()).item() for criterion in criteria])
        output_test = output_test.to("cpu")
        output_test = output_test.detach().numpy()
        output_test = output_test[:, 0]
        output_test = output_test * (maxT - minT) + minT
        target_test = target_test.to("cpu")
        target_test = target_test.detach().numpy()
        target_test = target_test[:, 0]
        target_test = target_test * (maxT - minT) + minT
        ypred.append(output_test)
        ylab.append(target_test)
        cumulative_error_profiles.append((output_test - target_test).cumsum())
    ypred, ylab = np.asarray(ypred), np.asarray(ylab)
    absolute_error = (ypred - ylab).mean(axis=1)
    losses = np.asarray(losses)
    def make_plot(arr, title):
        plt.hist(arr, bins=20, color="blue")
        plt.title(title)
        plt.axvline(arr.mean(), linestyle="--", color="orange")
        plt.savefig(reformat_filepath(f"{folder}/results/{filename}/{title}.png"))
        plt.close()
    make_plot(absolute_error, "error")
    make_plot(losses[:, 0], "MAE")
    make_plot(losses[:, 1], "MSE")
    plt.plot(cumulative_error_profiles)
    plt.title("Cumulative error profiles")
    plt.savefig(reformat_filepath(f"{folder}/results/{filename}/cumulative_error_profiles.png"))
    plt.close()
    for index, (prediction, target) in enumerate(zip(ypred, ylab)):
        plt.plot(prediction, label="pred")
        plt.plot(target, label="true")
        plt.legend()
        plt.title(f"Batch {index}")
        plt.savefig(reformat_filepath(f"{folder}/results/{filename}/profile_{index}.png"))
        plt.close()
    return {
        "absolute error": absolute_error.mean(),
        "MAE": losses[:, 0].mean(),
        "MSE": losses[:, 1].mean()
    }

def run(df, file_suffix=""):
    filename = f'{file_suffix}'
    os.makedirs(reformat_filepath(f"{folder}/results/{filename}"), exist_ok=True)
    os.makedirs(reformat_filepath(f"{folder}/results/{filename}/winter"), exist_ok=True)
    os.makedirs(reformat_filepath(f"{folder}/results/{filename}/spring"), exist_ok=True)
    os.makedirs(reformat_filepath(f"{folder}/results/{filename}/summer"), exist_ok=True)
    os.makedirs(reformat_filepath(f"{folder}/results/{filename}/autumn"), exist_ok=True)
    os.makedirs(reformat_filepath(f"{folder}/models/"), exist_ok=True)
    months = range(1, 13)
    data_dict = process_df(df, months)
    lstm = LSTM(
        n_features=data_dict["train"]["X"].shape[1],
        n_output=data_dict["train"]["y"].shape[1],
        seq_len=config["lb"],
        num_layers=config['num_layer'],
        num_hidden=config['hidden_size'],
        drop_prob=config['dropout'],
        weight_decay=config['weight_decay']
    ).to(config["device"])
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = getattr(torch.optim, config["optimizer_name"])(lstm.parameters(), lr=config["learning_rate"])    
    train(
        lstm, 
        data_dict["loaders"]["train"], 
        data_dict["loaders"]["val"],
        optimizer,
        criterion,
        config,
        data_dict["temp_limits"]["max"],
        data_dict["temp_limits"]["min"],
        filename
    )
    total_errors = eval(
        lstm,
        data_dict["loaders"]["test"],
        optimizer,
        data_dict["temp_limits"]["max"],
        data_dict["temp_limits"]["min"],
        filename
    )
    seasonal_errors = []
    for (index, loader) in enumerate(data_dict["loaders"]["test_by_season"]):
        seasonal_errors.append(
            eval(
                lstm,
                data_dict["loaders"]["test_by_season"][index],
                optimizer,
                data_dict["temp_limits"]["max"],
                data_dict["temp_limits"]["min"],
                reformat_filepath(f"{filename}/{['winter', 'spring', 'summer', 'autumn'][index]}")
            )
        )
    return total_errors, seasonal_errors