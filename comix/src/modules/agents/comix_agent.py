import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CEMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMAgent, self).__init__()
        self.args = args
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, inputs, actions):
        if actions is not None:
            inputs = th.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return {"Q": q}


class NAFAgent(nn.Module):
    # using code similar to https://github.com/ikostrikov/pytorch-ddpg-naf
    def __init__(self, input_shape, args):
        super(NAFAgent, self).__init__()
        self.args = args
        hidden_size = args.naf_hidden_dim
        num_inputs = input_shape
        num_outputs = args.n_actions

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = Variable(th.tril(th.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(th.diag(th.diag(
            th.ones(num_outputs, num_outputs))).unsqueeze(0))
        if self.args.use_cuda:
            self.tril_mask = self.tril_mask.cuda()
            self.diag_mask = self.diag_mask.cuda()

    def forward(self, inputs, actions=None):
        x, u = inputs, actions  # need to get to format bs*a, v

        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        mu = F.tanh(self.mu(x))

        if actions is not None:
            u = actions.contiguous().view(-1, actions.shape[-1])

            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + th.exp(L) * self.diag_mask.expand_as(L)
            P = th.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(-1)  # unsqueeze last dimension
            A = -0.5 * \
                th.bmm(th.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V
        else:
            actions = mu.detach()  # automatically pick best action
            Q = V

        return {"Q": Q,
                "V": V,
                "actions": actions}

    def get_weight_decay_weights(self):
        labels = ["V.weight", "V.bias", "L.weight", "L.bias", "mu.weight", "mu.bias"]
        named_params = dict(self.named_parameters())
        return {k:v for k, v in named_params.items() if k in labels}