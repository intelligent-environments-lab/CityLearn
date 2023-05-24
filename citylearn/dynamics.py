from pathlib import Path
from typing import List, Union
import torch
import torch.nn

class Dynamics:
    """Base building dynamics model."""

    def __init__(self):
        pass

    def reset(self):
        pass

class LSTMDynamics(Dynamics, torch.nn.Module):
    """LSTM building dynamics model that predicts indoor temperature based on partial cooling/heating load and other weather variables.
    
    Parameters
    ----------
    filepath: Union[Path, str]
        Path to model state dictionary.
    input_observation_names: List[str]
        List of maximum values used for input observation min-max normalization.
    input_normalization_minimum: List[float]
        List of minumum values used for input observation min-max normalization.
    input_normalization_maximum: List[float]
        List of maximum values used for input observation min-max normalization.
    input_size: int
        Number of variables used for prediction. This may not equal `input_observation_names`
        e.g. cooling and heating demand may be included in `input_observation_names` but only
        one of two may be used for the actual prediction depending on building needs.
    hidden_size: int
        The number of neurons in hidden layer.
    num_layers: int
        Number of hidden layers.
    lookback: int
        Number of samples used for prediction.
    """

    def __init__(
            self, filepath: Union[Path, str], input_observation_names: List[str], input_normalization_minimum: List[float], 
            input_normalization_maximum: List[float], input_size: int, hidden_size: int, num_layers: int, lookback: int
    ) -> None:
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
        self._hidden_state = None
        self._model_input = None
    
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
        """Loads dynamic model state dict, and initializes hidden states and model input."""

        super().reset()

        try:
            self.load_state_dict(torch.load(self.filepath)['model_state_dict'])
        except:
            self.load_state_dict(torch.load(self.filepath))

        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]

    def terminate(self):
        return