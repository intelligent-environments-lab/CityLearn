from torch import zeros
from torch.nn import Linear, LSTM, Module 

class LSTMBuildingDynamics(Module):
    def __init__(self, input_size: int = None, hidden_size: int = None, num_layers: int = None, batch_first: bool = None, lookback: int = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lookback = lookback
        self.lstm = self.set_lstm()
        self.linear = self.set_linear()

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @batch_first.setter
    def batch_first(self, batch_first: bool):
        self.__batch_first = True if batch_first is None else batch_first
    
    def set_lstm(self) -> LSTM:
        return LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

    def set_linear(self) -> Linear:
        return Linear(
            in_features=self.hidden_size,
            out_features=1
        )
    
    def forward(self, x, h):
        lstm_out, h = self.lstm(x, h)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        hidden_state = zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)
        return hidden

    def reset(self):
        return

    def terminate(self):
        return