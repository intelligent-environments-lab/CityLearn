import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.net = nn.ModuleList()
        for i in range(self.n_agents):
            self.net.append(
                nn.Sequential(
                    nn.Linear(input_shape, args.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.rnn_hidden_dim, args.n_actions)
                )
            )

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def forward(self, inputs, actions=None):
        inputs = inputs.reshape(-1, self.n_agents, inputs.shape[-1])

        actions = []
        for i in range(self.n_agents):
            action = self.net[i](inputs[:,i:i+1])
            if not self.agent_return_logits:
                action = th.tanh(action)
            actions.append(action)
        actions = th.cat(actions, 1).view(-1, self.args.n_actions)
        return {"actions": actions}
