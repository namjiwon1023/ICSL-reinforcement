import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, actor_dims, n_actions, name, args):
        super(ActorNetwork, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(actor_dims, self.args.n_hiddens_1)
        self.fc2 = nn.Linear(self.args.n_hiddens_1, self.args.n_hiddens_2)
        self.pi = nn.Linear(self.args.n_hiddens_2, n_actions)

        self.reset_parameters()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.args.actor_lr)
        self.to(self.args.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        out = self.pi(x)
        noise = T.rand_like(out)
        pi = F.softmax(out - T.log(-T.log(noise)), dim=-1) if self.args.explore else F.softmax(out, dim=-1)
        return pi


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.pi.weight, gain=nn.init.calculate_gain('leaky_relu'))