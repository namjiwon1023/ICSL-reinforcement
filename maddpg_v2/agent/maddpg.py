import torch as T
import torch.nn.functional as F
from agent.agent import Agent
from utils.ReplayBuffer import MultiAgentReplayBuffer

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, args):
        self.args = args
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, self.args))
        self.memory = MultiAgentReplayBuffer(critic_dims, actor_dims, n_actions, n_agents, self.args)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(obs[agent_idx])
            actions.append(action)

        return actions

    def learn(self):
        if not self.memory.ready():
            return

        if self.args.use_cuda:
            actor_states, states, actions, rewards, actor_next_states, next_states, dones = self.memory.sample_buffer(self.args.batch_size)
        else:
            actor_states, states, actions, rewards, actor_next_states, next_states, dones = self.memory.sample_buffer(self.args.batch_size)

            states = T.as_tensor(states, dtype=T.float, device=self.args.device)
            actions = T.as_tensor(actions, dtype=T.float, device=self.args.device)
            rewards = T.as_tensor(rewards, device=self.args.device)
            next_states = T.as_tensor(next_states, dtype=T.float, device=self.args.device)
            dones = T.as_tensor(dones, device=self.args.device)

        all_agents_next_actions = []
        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            if self.args.use_cuda:
                actor_next_states_batch = actor_next_states[agent_idx]
                next_action = agent.actor_target(actor_next_states_batch)
                all_agents_next_actions.append(next_action)

                actor_states_batch = actor_states[agent_idx]
                new_actions = agent.actor(actor_states_batch)
                all_agents_new_actions.append(new_actions)

                old_agents_actions.append(actions[agent_idx])

            else:
                actor_next_states_batch = T.as_tensor(actor_next_states[agent_idx], dtype=T.float, device=self.args.device)
                next_action = agent.actor_target(actor_next_states_batch)
                all_agents_next_actions.append(next_action)

                actor_states_batch = T.as_tensor(actor_states[agent_idx], dtype=T.float, device=self.args.device)
                new_actions = agent.actor(actor_states_batch)
                all_agents_new_actions.append(new_actions)

                old_agents_actions.append(actions[agent_idx])

        actor_next_actions = T.cat([acts for acts in all_agents_next_actions], dim=1)
        new_actor_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        curr_agents_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.critic_target(next_states, actor_next_actions).flatten()
                critic_value_[dones[:,0]] = 0.0
                target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_value = agent.critic(states, curr_agents_actions).flatten()
            critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic(states, new_actor_actions).flatten()
            actor_loss = -T.mean(actor_loss)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent._soft_update_target_network()