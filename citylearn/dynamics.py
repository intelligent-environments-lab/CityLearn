from pathlib import Path
from typing import List, Union
import torch
import torch.nn

class Dynamics:
    def __init__(self):
        """Intialize base `Dynamics`."""

        pass

class LSTMDynamics(Dynamics, torch.nn.Module):
    """Predicts indoor temperature based on partial cooling/heating load and other weather variables"""

    def __init__(
            self, filepath: Union[Path, str], input_normalization_minimum: List[float], input_normalization_maximum: List[float], 
            input_observation_names: List[str], input_size: int, hidden_size: int, num_layers: int, lookback: int
    ) -> None:
        """Intialize `LSTMDynamics`."""
        
        Dynamics.__init__(self)
        torch.nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.filepath = filepath
        self.input_observation_names = input_observation_names
        self.input_normalization_minimum = input_normalization_minimum
        self.input_normalization_maximum = input_normalization_maximum
        assert len(self.input_observation_names) == len(self.input_normalization_minimum) == len(self.input_normalization_maximum),\
            'input_observation_names, input_normalization_minimum and input_normalization_maximum must have the same length.'
        self.l_lstm = self.set_lstm()
        self.l_linear = self.set_linear()
    
    def set_lstm(self) -> torch.nn.LSTM:
        """Initialize LSTM model."""

        return torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def set_linear(self) -> torch.nn.Linear:
        """Initialize linear transformer."""

        return torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )
    
    def forward(self, x, h):
        """Predict indoor dry bulb temperature."""

        lstm_out, h = self.l_lstm(x, h)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size: int):
        """Initialize hidden states."""

        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)
        
        return hidden

    def reset(self):
        return

    def terminate(self):
        return