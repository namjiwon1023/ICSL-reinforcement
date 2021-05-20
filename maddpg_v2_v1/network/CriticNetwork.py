import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CriticNetwork(nn.Module):
    def __init__(self, critic_dims, n_agents, n_actions, name, args):
        super(CriticNetwork, self).__init__()
        self.args = args
        self.checkpoint_file = os.path.join(self.args.save_dir + '/' + self.args.scenario_name, name)

        self.fc1 = nn.Linear(critic_dims + n_agents * n_actions, self.args.n_hiddens_1)
        self.fc2 = nn.Linear(self.args.n_hiddens_1, self.args.n_hiddens_2)
        self.q = nn.Linear(self.args.n_hiddens_2, 1)

        self.apply(self._layer_norm)

        self.optimizer = optim.Adam(self.parameters(), lr=self.args.critic_lr)

        self.to(self.args.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _layer_norm(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)