import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_features: int, n_output: int, drop_prob: float, seq_len: int, num_hidden: int, num_layers: int, weight_decay: float):
        super(LSTM, self).__init__()

        self.n_features = n_features  # number of inputs variable
        self.n_output = n_output  # number of output
        self.seq_len = seq_len  # lookback value
        self.n_hidden = num_hidden  # number of hidden states
        self.n_layers = num_layers  # number of LSTM layers (stacked)
        self.weight_decay = weight_decay

        self.l_lstm = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.n_hidden,
                              num_layers=self.n_layers,
                              batch_first=True)
        self.dropout = torch.nn.Dropout(drop_prob)
        # LSTM Outputs: output, (h_n, c_n)
        # according to pytorch docs LSTM output is
        # (batch_size, seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden, n_output)

        # Add L2 regularization to the linear layer
        self.l_linear.weight_decay = self.weight_decay

    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains the same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

    def forward(self, input_tensor, hidden_cell_tuple):
        batch_size, seq_len, _ = input_tensor.size()
        lstm_out, hidden_cell_tuple = self.l_lstm(input_tensor, hidden_cell_tuple)
        lstm_out = self.dropout(lstm_out)  # Applying dropout
        out = lstm_out[:, -1, :]  # I take only the last output vector, for each Batch
        out_linear = self.l_linear(out)
        return out_linear, hidden_cell_tuple