from typing import Any, Mapping, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from citylearn.end_use_load_profiles.lstm_model.model import LSTM
from citylearn.end_use_load_profiles.lstm_model.preprocessing import preprocess_df

def run(config: Mapping[str, Any], df: pd.DataFrame) -> Tuple[LSTM, Mapping[str, Any], Mapping[str, float]]:
    # preprocess data
    data_dict = preprocess_df(config, df)

    # initialize LSTM model
    lstm = LSTM(
        n_features=data_dict['train']['X'].shape[1],
        n_output=data_dict['train']['y'].shape[1],
        seq_len=config['lb'],
        num_layers=config['num_layer'],
        num_hidden=config['hidden_size'],
        drop_prob=config['dropout'],
        weight_decay=config['weight_decay']
    ).to(config['device'])
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = getattr(torch.optim, config['optimizer_name'])(lstm.parameters(), lr=config['learning_rate'])    
    
    # train model
    lstm = train(
        lstm, 
        data_dict['loaders']['train'], 
        data_dict['loaders']['val'],
        optimizer,
        criterion,
        config,
        data_dict['temp_limits']['max'],
        data_dict['temp_limits']['min'],
    )

    # get model error summary
    total_errors = eval(
        config,
        lstm,
        data_dict['loaders']['test'],
        optimizer,
        data_dict['temp_limits']['max'],
        data_dict['temp_limits']['min'],
    )
    
    return lstm, data_dict['observation_metadata'], total_errors

def eval(config: Mapping[str, Any], model: LSTM, test_loader: DataLoader, optimizer: torch.optim.Optimizer, temperature_normalization_maximum: float, temperature_normalization_minimum: float) -> Mapping[str, float]:
    model.eval()
    h = model.init_hidden(config['batch_size'], config['device'])
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
        input_test = input_test.to(config['device'])
        target_test = target_test.to(config['device'])
        h = tuple([each.data for each in h])
        output_test, h = model(input_test.float(), h)
        optimizer.zero_grad()
        losses.append([criterion(output_test, target_test.float()).item() for criterion in criteria])
        output_test = output_test.to('cpu')
        output_test = output_test.detach().numpy()
        output_test = output_test[:, 0]
        output_test = output_test * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum
        target_test = target_test.to('cpu')
        target_test = target_test.detach().numpy()
        target_test = target_test[:, 0]
        target_test = target_test * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum
        ypred.append(output_test)
        ylab.append(target_test)
        cumulative_error_profiles.append((output_test - target_test).cumsum())
    
    ypred, ylab = np.asarray(ypred), np.asarray(ylab)
    absolute_error = (ypred - ylab).mean(axis=1)
    losses = np.asarray(losses)

    return {
        'absolute error': absolute_error.mean(),
        'mae': losses[:, 0].mean(),
        'rmse': losses[:, 1].mean()
    }

def train(lstm: LSTM, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.MSELoss, config: Mapping[str, Any], temperature_normalization_maximum: float, temperature_normalization_minimum: float) -> LSTM:
    lstm.train()
    data = []
    train_losses, val_losses = [], []
    
    for _ in range(config['epochs']):
        loss_train, loss_val, \
        ylab_train, ypred_train, \
        ylab_val, ypred_val = _train(model=lstm, train_loader=train_loader, val_loader=val_loader,
                                        optimizer=optimizer, criterion=criterion, config=config,
                                        temperature_normalization_maximum=temperature_normalization_maximum, temperature_normalization_minimum=temperature_normalization_minimum)
        data.append([np.asarray(ylab_val), np.asarray(ypred_val)])
        train_losses.append(loss_train)
        val_losses.append(loss_val)

    return lstm

# TODO: Explicitly define the data type contained in returned lists
def _train(model: LSTM, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.MSELoss, config: Mapping[str, Any], temperature_normalization_maximum: float, temperature_normalization_minimum: float, train_loss_list: list = None, val_loss_list: list = None) -> Tuple[list, list, list, list, list, list]:
    ypred_train = []
    ylab_train = []
    ypred_val = []
    ylab_val = []
    train_loss_list = [] if train_loss_list is None else train_loss_list
    val_loss_list = [] if val_loss_list is None else val_loss_list

    model.train()
    # Hidden state is initialized at each epochs
    h = model.init_hidden(config["batch_size"], config["device"])
    for batch in train_loader:
        input_train, target_train = batch
        input_train = input_train.to(config["device"])
        target_train = target_train.to(config["device"])

        # since the batch is big enough, a stateless mode is used
        # (also considering the possibility to shuffle the training examples,
        # which increase the generalization ability of the network)
        h = model.init_hidden(config["batch_size"], config["device"])
        h = tuple([each.data for each in h])

        # FORWARD PASS
        output_train, h = model(input_train.float(), h)
        loss_train = criterion(output_train, target_train.float())

        # BACKWARD PASS
        optimizer.zero_grad()
        # obtain the loss function
        loss_train.backward()

        # STEP WITH OPTIMIZER
        optimizer.step()

        # EVALUATE METRICS FOR TRAIN
        # if config.device.type == 'cuda':
        output_train = output_train.to('cpu')
        output_train = output_train.detach().numpy()
        # RESCALE OUTPUT
        output_train = output_train[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_train = output_train * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_train = target_train.to('cpu')
        target_train = target_train.detach().numpy()
        target_train = target_train[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_train = target_train * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum
        ypred_train.append(output_train)
        ylab_train.append(target_train)

    # train_loss_list.append(loss_train.item())
    train_loss_list = loss_train.item()

    flatten = lambda l: [item for sublist in l for item in sublist]
    ypred_train = flatten(ypred_train)
    ylab_train = flatten(ylab_train)

    h = model.init_hidden(config["batch_size"], config["device"])
    for batch in val_loader:
        input_val, target_val = batch
        input_val = input_val.to(config["device"])
        target_val = target_val.to(config["device"])

        h = tuple([each.data for each in h])
        output_val, h = model(input_val.float(), h)
        # target_val = target_val.unsqueeze(1)

        optimizer.zero_grad()

        # obtain loss function
        loss_val = criterion(output_val, target_val.float())

        # EVALUATE METRICS FOR TRAIN
        # if config.device.type == 'cuda':
        output_val = output_val.to('cpu')
        output_val = output_val.detach().numpy()
        # RESCALE OUTPUT
        output_val = output_val[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_val = output_val * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_val = target_val.to('cpu')
        target_val = target_val.detach().numpy()
        target_val = target_val[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_val = target_val * (temperature_normalization_maximum - temperature_normalization_minimum) + temperature_normalization_minimum
        ypred_val.append(output_val)
        ylab_val.append(target_val)

    # val_loss_list.append(loss_val.item())
    val_loss_list = loss_val.item()

    # if len(train_loss_list) >= (config.epochs - 1):
    #     train_loss_list = []
    # if len(val_loss_list) >= (config.epochs - 1):
    #     val_loss_list = []

    return train_loss_list, val_loss_list, ylab_train, ypred_train, ylab_val, ypred_val