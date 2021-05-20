import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ActorNetwork(nn.Module):
    def __init__(self, actor_dims, n_actions, name, args):
        super(ActorNetwork, self).__init__()
        self.args = args

        self.checkpoint_file = os.path.join(self.args.save_dir + '/' + self.args.scenario_name, name)

        self.fc1 = nn.Linear(actor_dims, self.args.n_hiddens_1)
        self.fc2 = nn.Linear(self.args.n_hiddens_1, self.args.n_hiddens_2)
        self.pi = nn.Linear(self.args.n_hiddens_2, n_actions)

        self.apply(self._layer_norm)

        self.optimizer = optim.Adam(self.parameters(), lr=self.args.actor_lr)

        self.to(self.args.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _layer_norm(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)