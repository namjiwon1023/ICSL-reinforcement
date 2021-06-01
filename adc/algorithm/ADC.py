import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import gym
# from gym.wrappers import RescaleAction
import random

from utils.ReplayBuffer import ReplayBuffer
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork


class ADCAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.env = gym.make('Pendulum-v0')
        # self.env = RescaleAction(self.env, -1, 1)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.n_hiddens = int(2**6)

        self.memory = ReplayBuffer(self.memory_size, self.n_states, self.batch_size)

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.n_hiddens, self.actor_lr, self.device, self.max_action)
        self.critic = CriticNetwork(self.n_states, self.n_actions, self.n_hiddens, self.critic_lr, self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False


        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.transition = list()

        self.dirPath = os.getcwd() + '/' + 'actor_parameters.pth'
        if os.path.exists(self.dirPath):
            self.load_models()
        else:
            print('|------------------------------------|')
            print('|----- No parameters available! -----|')
            print('|------------------------------------|')



    def choose_action(self, state, test_mode=False):
        if test_mode is True:
            with T.no_grad():
                action = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                action = action.detach().cpu().numpy()
        else:
            if self.epsilon > np.random.random():
                action = self.env.action_space.sample()
            else:
                with T.no_grad():
                    action = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                    action = action.detach().cpu().numpy()

        self.transition = [state, action]
        return action

    def target_soft_update(self):
        tau = self.tau
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)
            for t_p, l_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)


    def learn(self):
        value_losses,  Policy_losses = 0, 0
        for e in range(self.gradient_steps):
            self.learning_steps += 1
            with T.no_grad():
                samples = self.memory.sample_batch(self.batch_size)
                state = T.as_tensor(samples["state"], device=self.device)
                next_state = T.as_tensor(samples["next_state"], device=self.device)
                action = T.as_tensor(samples["action"].reshape(-1, 1), device=self.device)
                reward = T.as_tensor(samples["reward"].reshape(-1, 1), device=self.device)
                mask = T.as_tensor(samples["mask"].reshape(-1, 1), device=self.device)

                # critic update
                next_action = self.actor_target(next_state)
                next_q_value = self.critic_target(next_state, next_action)
                value_target = reward + next_q_value * mask
            q_eval = self.critic(state, action)

            critic_loss = F.mse_loss(q_eval, value_target)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor.optimizer.zero_grad()
            actor_step_loss.backward()
            self.actor.optimizer.step()

            self.epsilon = max(0.05, self.epsilon - 0.0000005)


            value_losses += critic_loss.detach().item()
            Policy_losses += actor_loss.detach().item()


            if e % self.target_update_interval == 0:
                self.target_soft_update()

        return value_losses, Policy_losses, self.epsilon

    def save_models(self):

        print('-------------------------')
        print('------ Save models ! ----')
        print('-------------------------')

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()


    def load_models(self):

        print('-------------------------')
        print('------ Load models ! ----')
        print('-------------------------')

        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                action = self.choose_action(state, test_mode=True)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / n_starts