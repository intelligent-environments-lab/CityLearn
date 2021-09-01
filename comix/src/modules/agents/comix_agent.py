import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CEMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        num_inputs = input_shape + args.n_actions
        self.num_inputs = num_inputs
        hidden_size = args.rnn_hidden_dim

        self.net = nn.ModuleList()
        for i in range(self.n_agents):
            self.net.append(
                nn.Sequential(
                    nn.Linear(num_inputs, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            )
            self.net[-1][-1].weight.data.uniform_(-3e-3, 3e-3)
            self.net[-1][-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, inputs, actions):
        if actions is not None:
            inputs = th.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        inputs = inputs.reshape(-1, self.n_agents, self.num_inputs)
        qs = []
        for i in range(self.n_agents):
            qs.append(self.net[i](inputs[:,i:i+1]))
        q = th.cat(qs, 1)
        q = q.view(-1, 1)
        return {"Q": q}


class NAFAgent(nn.Module):
    # using code similar to https://github.com/ikostrikov/pytorch-ddpg-naf
    def __init__(self, input_shape, args):
        super(NAFAgent, self).__init__()
        self.args = args
        hidden_size = args.naf_hidden_dim
        num_inputs = input_shape
        self.num_inputs = num_inputs
        self.n_agents = args.n_agents
        num_outputs = args.n_actions

        self.net = nn.ModuleList()
        self.Vs = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.Ls = nn.ModuleList()
        for i in range(self.n_agents):
            self.net.append(
                nn.Sequential(
                    nn.Linear(num_inputs, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size)
                )
            )
            self.Vs.append(nn.Linear(hidden_size, 1))
            self.Vs[-1].weight.data.mul_(0.1)
            self.Vs[-1].bias.data.mul_(0.1)

            self.mus.append(nn.Linear(hidden_size, num_outputs))
            self.mus[-1].weight.data.mul_(0.1)
            self.mus[-1].bias.data.mul_(0.1)

            self.Ls.append(nn.Linear(hidden_size, num_outputs ** 2))
            self.Ls[-1].weight.data.mul_(0.1)
            self.Ls[-1].bias.data.mul_(0.1)

        self.tril_mask = Variable(th.tril(th.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(th.diag(th.diag(
            th.ones(num_outputs, num_outputs))).unsqueeze(0))
        if self.args.use_cuda:
            self.tril_mask = self.tril_mask.cuda()
            self.diag_mask = self.diag_mask.cuda()

    def forward(self, inputs, actions=None):
        x, u = inputs, actions  # need to get to format bs*a, v
        x = x.view(-1, self.n_agents, x.shape[-1])
        if u is not None:
            u = u.view(-1, self.n_agents, u.shape[-1])

        V = []
        mu = []
        L = []
        for i in range(self.n_agents):
            xi = self.net[i](x[:,i:i+1])
            vi = self.Vs[i](xi)
            mui = self.mus[i](xi)
            li = self.Ls[i](xi)
            V.append(vi)
            mu.append(mui)
            L.append(li)

        V = th.cat(V, 1).view(-1, V[0].shape[-1])
        mu = th.cat(mu, 1).view(-1, mu[0].shape[-1])
        num_outputs = mu.size(1)
        L = th.cat(L, 1).view(-1, num_outputs, num_outputs)

        if actions is not None:
            u = actions.contiguous().view(-1, actions.shape[-1])
            #num_outputs = mu.size(1)
            #L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * self.tril_mask.expand_as(L) + th.exp(L) * self.diag_mask.expand_as(L)
            P = th.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(-1)  # unsqueeze last dimension
            A = -0.5 * th.bmm(th.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]
            Q = A + V
        else:
            actions = mu.detach()  # automatically pick best action
            Q = V

        return {"Q": Q,
                "V": V,
                "actions": actions}

    def get_weight_decay_weights(self):
        #labels = ["Vs.0.weight", "V.bias", "L.weight", "L.bias", "mu.weight", "mu.bias"]
        named_params = dict(self.named_parameters())
        labels = []
        for label in named_parameters.keys():
            if "Vs" in label or "Ls" in label or "mus" in label:
                labels.append(label)
        return {k:v for k, v in named_params.items() if k in labels}
