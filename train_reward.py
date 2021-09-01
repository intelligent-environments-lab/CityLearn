import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

class SAReward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, c_dim):
        super(SAReward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_dim))

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.net(x)

class Reward(nn.Module):
    def __init__(self, n_agents, state_dims, action_dims, hidden_dim, c_dim):
        super(Reward, self).__init__()
        self.n_agents = n_agents
        self.single_rew = nn.ModuleList()
        for i, sd, ad in zip(list(range(n_agents)), state_dims, action_dims):
            self.single_rew.append(SAReward(sd, ad, hidden_dim, c_dim))

    def forward(self, s, a):
        """
        s: list [batch, time, state_dim]
        a: list [batch, time, action_dim]
        """
        r = sum([self.single_rew[i](s[i], a[i]).mean(1) for i in range(self.n_agents)])
        return r / self.n_agents

class RewData(Dataset):
    def __init__(self, data):
        self.s = data['s']
        self.a = data['a']
        self.c = data['c']
        self.n = len(self.c)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.c[idx]

def train():
    data = torch.load("data1.pt")
    costs = data['costs']
    osp = data['observation_space']
    asp = data['action_space']
    rewdata = RewData(data['data'])
    loader = torch.utils.data.DataLoader(rewdata, batch_size=2, num_workers=1, shuffle=True)

    reward_fn = Reward(9, [osp[i].low.shape[0] for i in range(9)], [asp[i].low.shape[0] for i in range(9)], 128, len(costs))
    reward_fn.cuda()
    optimizer = torch.optim.Adam(reward_fn.parameters(), lr=1e-3, weight_decay=1e-5)

    for ep in range(1000):
        L = 0
        for s,a,c in loader:
            s = [s[i].cuda() for i in range(9)]
            a = [a[i].cuda() for i in range(9)]
            c = c.cuda()
            c_ = reward_fn(s,a)
            optimizer.zero_grad()
            loss = F.mse_loss(c_, c)
            loss.backward()
            optimizer.step()
            L += loss.item()
        print(ep, L)

train()
