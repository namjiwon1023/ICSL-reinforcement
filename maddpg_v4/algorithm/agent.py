import torch as T
import os
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
import numpy as np


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.device = self.args.device
        self.agent_name = 'agent_%s' % agent_id

        self.actor = ActorNetwork(args, agent_id, self.agent_name + '_actor')
        self.critic = CriticNetwork(args, self.agent_name + '_critic_target')

        self.actor_target = ActorNetwork(args, agent_id, self.agent_name+ '_actor_target')
        self.critic_target = CriticNetwork(args, self.agent_name + '_critic_target')

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def choose_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            state = T.tensor(o, dtype=T.float32).unsqueeze(0).to(self.device)
            pi = self.actor(state).squeeze(0)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u

    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

