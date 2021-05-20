import torch as T
import torch.nn.functional as F
from agent.agent import Agent
from utils.ReplayBuffer import MultiAgentReplayBuffer
import copy

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, args):
        self.args = args
        self.gamma = self.args.gamma
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.memory = MultiAgentReplayBuffer(critic_dims, actor_dims, n_actions, n_agents, self.args)

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, self.args))

    def choose_action(self, obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(obs[agent_idx])
            actions.append(action)

        return actions

    def learn(self):
        if not self.memory.ready():
            return

        for agent_idx in range(self.n_agents):
            actor_states, states, actions, rewards, actor_next_states, next_states, dones = self.memory.sample_buffer(self.args.batch_size)

            batch_states = T.as_tensor(states, dtype=T.float32, device=self.args.device)
            batch_actions = T.as_tensor(actions, dtype=T.float32, device=self.args.device)
            batch_rewards = T.as_tensor(rewards, device=self.args.device)
            batch_next_states = T.as_tensor(next_states, dtype=T.float32, device=self.args.device)
            batch_dones = T.as_tensor(dones, dtype=T.float32, device=self.args.device)


            curr_actions_list = []
            for i in range(self.n_agents):
                curr_actions_list.append(batch_actions[i])
            curr_actions = T.cat([acts for acts in curr_actions_list], dim=1)

            with T.no_grad():
                target_next_actions = T.cat([self.agents[n].actor_target.forward(T.as_tensor(actor_next_states[n], dtype=T.float32, device=self.args.device)) for n in range(self.n_agents)], dim=1)
                target_next_value = self.agents[agent_idx].critic_target.forward(batch_next_states, target_next_actions).view(-1)
                target = batch_rewards[:,agent_idx] + self.gamma * (1 - batch_dones[:,agent_idx]) * target_next_value
            q_value = self.agents[agent_idx].critic.forward(batch_states, curr_actions).view(-1)
            critic_loss = (target - q_value).pow(2).mean()

            self.agents[agent_idx].critic.optimizer.zero_grad()
            critic_loss.backward()
            self.agents[agent_idx].critic.optimizer.step()


            new_action = self.agents[agent_idx].actor.forward(T.as_tensor(actor_states[agent_idx], dtype=T.float32, device=self.args.device))
            new_actions_list = copy.deepcopy(curr_actions_list)
            new_actions_list[agent_idx] = new_action
            new_actions = T.cat([acts for acts in new_actions_list], dim=1)

            actor_loss = self.agents[agent_idx].critic.forward(batch_states, new_actions)
            actor_loss = -T.mean(actor_loss)

            self.agents[agent_idx].actor.optimizer.zero_grad()
            actor_loss.backward()
            self.agents[agent_idx].actor.optimizer.step()

            self.agents[agent_idx]._soft_update_target_network()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
