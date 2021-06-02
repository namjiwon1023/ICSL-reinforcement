import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, alpha, device):
        super(CriticNetwork, self).__init__()
        chkpt_dir = os.getcwd()
        self.device = device
        self.checkpoint = os.path.join(chkpt_dir, 'critic_parameters.pth')

        self.feature = nn.Sequential(nn.Linear(n_states, 30),
                                    nn.ReLU(),
                                    nn.Linear(30, 30),
                                    nn.ReLU()
                                    )

        self.state_value = nn.Sequential(nn.Linear(20, 1))


        self.advantage = nn.Sequential(nn.Linear(30, 100))

        self.reset_parameters(self.feature)
        self.reset_parameters(self.state_value)
        self.reset_parameters(self.advantage)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)


    def forward(self, state):
        feature = self.feature(state)

        state_value = self.state_value(feature)

        advantage = self.advantage(feature)

        Q_value = state_value + advantage - advantage.mean(dim=-1, keepdim=True)

        return Q_value


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