import numpy as np
import torch as T
import os
from algorithm.agent import Agent
from utils.ReplayBuffer import ReplayBuffer


class MADDPG:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.memory = ReplayBuffer(args)

        self.agents = []
        for agent_idx in range(self.args.n_agents):
            agent = Agent(agent_idx, self.args)
            self.agents.append(agent)

    def choose_action(self, o, noise_rate, epsilon):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(o[agent_idx], noise_rate, epsilon)
            actions.append(action)
        return actions

    def learn(self):
        if not self.memory.ready():
            return
        for agent_idx in range(self.args.n_agents):

            transitions = self.memory.sample_batch(self.args.batch_size)

            for key in transitions.keys():
                transitions[key] = T.tensor(transitions[key], dtype=T.float32).to(self.device)

            observations, actions, next_observations = [], [], []
            for n in range(self.args.n_agents):
                observations.append(transitions['observations_%d' % n])
                actions.append(transitions['actions_%d' % n])
                next_observations.append(transitions['next_observations_%d' % n])

            next_actions = []
            with T.no_grad():
                for i in range(self.args.n_agents):
                        next_actions.append(self.agents[i].actor_target(next_observations[i]))

            reward = transitions['rewards_%d' % agent_idx]
            with T.no_grad():
                q_next = self.agents[agent_idx].critic_target(next_observations, next_actions).detach()
                target_q = (reward.unsqueeze(1) + self.args.gamma * q_next).detach()
            q_value = self.agents[agent_idx].critic(observations, actions)
            critic_loss = (target_q - q_value).pow(2).mean()

            actions[agent_idx] = self.agents[agent_idx].actor(observations[agent_idx])
            actor_loss = -self.agents[agent_idx].critic(observations, actions).mean()

            self.agents[agent_idx].actor.optimizer.zero_grad()
            actor_loss.backward()
            self.agents[agent_idx].actor.optimizer.step()
            self.agents[agent_idx].critic.optimizer.zero_grad()
            critic_loss.backward()
            self.agents[agent_idx].critic.optimizer.step()

            self.agents[agent_idx]._soft_update_target_network()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
