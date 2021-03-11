import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy

class QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(QNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DQN')

        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, act_dim))

    def forward(self, state):
        out = self.critic(state)

        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class DoubleQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(DoubleQNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DoubleDQN')

        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, act_dim))

    def forward(self, state):
        out = self.critic(state)
        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class DuelingDQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(DuelingDQNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DuelingDQN')

        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU())

        self.advantage = nn.Sequential(nn.Linear(hidden_dim, act_dim))
        self.value = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, state):
        feature = self.feature(state)

        advantage = self.advantage(feature)
        value = self.value(feature)
        out = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class D3QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(D3QNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'D3QN')

        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU())

        self.advantage = nn.Sequential(nn.Linear(hidden_dim, act_dim))
        self.value = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, state):
        feature = self.feature(state)

        advantage = self.advantage(feature)
        value = self.value(feature)
        out = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))


class Dueling_Double_DQN_2Net(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(Dueling_Double_DQN_2Net, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'Dueling_Double_DQN_2Net')

        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU())

        self.advantage_1 = nn.Sequential(nn.Linear(hidden_dim, act_dim))
        self.advantage_2 = nn.Sequential(nn.Linear(hidden_dim, act_dim))

        self.value_1 = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.value_2 = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, state):
        feature = self.feature(state)

        advantage_1 = self.advantage_1(feature)
        value_1 = self.value_1(feature)
        q = value_1 + advantage_1 - advantage_1.mean(dim=-1, keepdim=True)

        return q

    def get_double_Q(self, state):
        feature = self.feature(state)

        advantage_1 = self.advantage_1(feature)
        value_1 = self.value_1(feature)
        q1 = value_1 + advantage_1 - advantage_1.mean(dim=-1, keepdim=True)

        advantage_2 = self.advantage_2(feature)
        value_2 = self.value_2(feature)
        q2 = value_2 + advantage_2 - advantage_2.mean(dim=-1, keepdim=True)

        return q1, q2

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))


'''
If you want to use a neural network(noise), please initialize it last

def reset_noise(self):
    self.NoisyLinearName.reset_noise()

'''
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = 0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))    # p : input , q: output ,size (q,p)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features,in_features))  # p : input , q: output ,size (q,p)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features,in_features))  # weight epsilon replay buffer in module

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))      # p : input , q: output ,size (q)
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))   # p : input , q: output ,size (q)
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))    # bias epsilon replay buffer in module

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range,mu_range)    # +- input.sqrt()
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(x,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon
                        )

    @staticmethod
    def scale_noise(size):
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())    # sign() : x > 0 : 1 , x = 0 : 0 , x < 0 : -1
