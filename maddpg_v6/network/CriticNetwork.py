import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, critic_dims, n_agents, n_actions, name, args):
        super(CriticNetwork, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(critic_dims + n_agents * n_actions, self.args.n_hiddens_1)
        self.fc2 = nn.Linear(self.args.n_hiddens_1, self.args.n_hiddens_2)
        self.q = nn.Linear(self.args.n_hiddens_2, 1)

        self.reset_parameters()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.args.critic_lr)
        self.to(self.args.device)

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.leaky_relu(self.fc2(x))
        q = self.q(x)
        return q


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q.weight, gain=nn.init.calculate_gain('leaky_relu'))