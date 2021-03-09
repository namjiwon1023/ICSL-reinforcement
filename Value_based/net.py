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
    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))

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
    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))

class D3QNetwork(nn.Module):
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
    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))

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
    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))