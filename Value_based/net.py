import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy

class QNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(QNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DQN')

        self.critic = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_actions))

    def forward(self, state):
        out = self.critic(state)

        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class DoubleDQNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(DoubleDQNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DoubleDQN')

        self.critic = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_actions))

    def forward(self, state):
        out = self.critic(state)
        return out

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class DuelingNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(DuelingNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DuelingDQN')

        self.feature = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU())

        self.advantage = nn.Sequential(nn.Linear(n_hidden, n_actions))
        self.value = nn.Sequential(nn.Linear(n_hidden, 1))

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
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(D3QNetwork, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'D3QN')

        self.feature = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU())

        self.advantage = nn.Sequential(nn.Linear(n_hidden, n_actions))
        self.value = nn.Sequential(nn.Linear(n_hidden, 1))

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

class DoubleDQN_2net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(DoubleDQN_2net, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'DoubleDQN_2net')

        self.critic1 = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_actions))

        self.critic2 = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_actions))

    def forward(self, state):
        q = self.critic1(state)

        return q

    def get_double_Q(self, state):
        q1 = self.critic1(state)
        q2 = self.critic2(state)

        return q1, q2


    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

class Dueling_Double_DQN_2Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, alpha,
                dirPath='/home/nam/ICSL-reinforcement/Value_based'):
        super(Dueling_Double_DQN_2Net, self).__init__()

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint = os.path.join(dirPath, 'Dueling_Double_DQN_2Net')

        self.feature = nn.Sequential(nn.Linear(n_states, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU())

        self.advantage_1 = nn.Sequential(nn.Linear(n_hidden, n_actions))
        self.advantage_2 = nn.Sequential(nn.Linear(n_hidden, n_actions))

        self.value_1 = nn.Sequential(nn.Linear(n_hidden, 1))
        self.value_2 = nn.Sequential(nn.Linear(n_hidden, 1))

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


class NoisyLinear(nn.Module):
    '''
    If you want to use a neural network(noise), please initialize it last
    def reset_noise(self):
        self.NoisyLinearName.reset_noise()
    '''
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


# Convolutional neural network structure
class CNN_1(nn.Module):
    '''
        input: [None, 3, 64, 64]; output: [None, 1024] -> [None, 512];
    '''
    def __init__(self, input_dims, n_actions):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 32, kernel_size = 8, stride = 4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)       # 64 -> 128 / 64 -> 64

        self.fc1 = nn.Linear(64*4*4, 512)

        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*4*4)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class CNN_2(nn.Module):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 4, 84, 84]; output: [None, 2592] -> [None, 256];
    '''
    def __init__(self, input_dims, n_actions):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)

        self.fc1 = nn.Linear(None, 256)    # Need to do the calculations yourself
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = x.view(-1, None)     # Need to do the calculations yourself

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class CNN_3(nn.Module):
    ''' DQN Nature 2015 paper
        input: [None, 4, 84, 84]; output: [None, 3136] -> [None, 512];
    '''
    def __init__(self, input_dims, n_actions):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(None, 512)    # Need to do the calculations yourself
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, None)    # Need to do the calculations yourself

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class CNN_4(nn.Module):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 2, 120, 160]; output: [None, 1280] -> [None, 256];
    '''
    def __init__(self, input_dims, n_actions):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 8, kernel_size = 5, stride = 4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2)

        self.fc1 = nn.Linear(None, 256)    # Need to do the calculations yourself
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, None)    # Need to do the calculations yourself

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out