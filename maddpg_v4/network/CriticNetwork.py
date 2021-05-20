import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, args, name):
        super(CriticNetwork, self).__init__()
        self.args = args
        self.checkpoint_file = os.path.join(self.args.save_dir + '/' + self.args.scenario_name, name)
        self.max_action = args.high_action
        self.device = args.device
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.n_hiddens_1)
        self.fc2 = nn.Linear(args.n_hiddens_1, args.n_hiddens_2)
        self.fc3 = nn.Linear(args.n_hiddens_2, args.n_hiddens_2)
        self.q_out = nn.Linear(args.n_hiddens_2, 1)
        self.apply(self._layer_norm)
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.to(self.device)

    def forward(self, state, action):
        state = T.cat(state, dim=1)

        for i in range(len(action)):
            action[i] /= self.max_action

        action = T.cat(action, dim=1)
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _layer_norm(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)