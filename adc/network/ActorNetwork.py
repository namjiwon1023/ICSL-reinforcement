import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, alpha, device, max_action):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.device = device
        chkpt_dir = os.getcwd()
        self.checkpoint = os.path.join(chkpt_dir, 'actor_parameters.pth')

        self.actor = nn.Sequential(nn.Linear(n_states, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, n_actions))

        self.reset_parameters(self.actor)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)


    def forward(self, state):
        action = self.actor(state)
        out = T.tanh(action) * self.max_action

        return out


    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def single_init(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint))