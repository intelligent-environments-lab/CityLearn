import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FacMADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FacMADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        #self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        #self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #self.fc3 = nn.Linear(args.rnn_hidden_dim, getattr(self.args, "q_embed_dim", 1))
        self.net = nn.ModuleList()
        for i in range(self.n_agents):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.input_shape, args.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(args.rnn_hidden_dim),
                    nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(args.rnn_hidden_dim),
                    nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.rnn_hidden_dim, 1)
                )
            )
            self.net[-1][-1].weight.data.uniform_(-3e-3, 3e-3)
            self.net[-1][-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        inputs = inputs.reshape(-1, self.n_agents, inputs.shape[-1])
        q = []
        for i in range(self.n_agents):
            qq = self.net[i](inputs[:,i:i+1])
            q.append(qq)
        q = th.cat(q, 1).view(-1, 1)
        #x = F.relu(self.fc1(inputs))
        #x = F.relu(self.fc2(x))
        #q = self.fc3(x)
        #q = self.net(inputs)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape
