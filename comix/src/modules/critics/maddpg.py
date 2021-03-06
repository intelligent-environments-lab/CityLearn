import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions * self.n_agents),
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the full state/joint obs input
        input_shape = scheme["state"]["vshape"]
        return input_shape