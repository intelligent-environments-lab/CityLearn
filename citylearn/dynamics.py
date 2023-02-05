import os
from pathlib import Path
from typing import Union
import torch
import torch.nn

class Dynamics:
    def __init__(self, root_directory: Union[str, Path]):
        self.root_directory = root_directory

class LSTMDynamics(Dynamics, torch.nn.Module):
    def __init__(self, *args, state_dict_filename: str, input_size: int = None, hidden_size: int = None, num_layers: int = None, batch_first: bool = None, lookback: int = None) -> None:
        Dynamics.__init__(self, *args)
        torch.nn.Module.__init__(self)
        self.state_dict_filename = state_dict_filename
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lookback = lookback
        self.l_lstm = self.set_lstm()
        self.l_linear = self.set_linear()
        self.load_state_dict(torch.load(os.path.join(self.root_directory, self.state_dict_filename)))
        self.hidden_list = self.init_hidden(1)

    @property
    def state_dict_filename(self) -> str:
        return self.__state_dict_filename

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @state_dict_filename.setter
    def state_dict_filename(self, state_dict_filename: str):
        self.__state_dict_filename = state_dict_filename

    @batch_first.setter
    def batch_first(self, batch_first: bool):
        self.__batch_first = True if batch_first is None else batch_first
    
    def set_lstm(self) -> torch.nn.LSTM:
        return torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

    def set_linear(self) -> torch.nn.Linear:
        return torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )
    
    def forward(self, x, h):
        lstm_out, h = self.l_lstm(x, h)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)
        return hidden

    def reset(self):
        return

    def terminate(self):
        return