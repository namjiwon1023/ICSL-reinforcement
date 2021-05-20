import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class ActorNetwork(nn.Module):
    def __init__(self, args, agent_id, name):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.checkpoint_file = os.path.join(self.args.save_dir + '/' + self.args.scenario_name, name)
        self.device = args.device
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], args.n_hiddens_1)
        self.fc2 = nn.Linear(args.n_hiddens_1, args.n_hiddens_2)
        self.fc3 = nn.Linear(args.n_hiddens_2, args.n_hiddens_2)
        self.action_out = nn.Linear(args.n_hiddens_2, args.action_shape[agent_id])
        self.apply(self._layer_norm)
        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * T.tanh(self.action_out(x))

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _layer_norm(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)