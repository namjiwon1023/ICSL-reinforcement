import torch as T
import torch.nn.functional as F
from agent.agent import Agent
from utils.ReplayBuffer import MultiAgentReplayBuffer
import copy
import os
import time

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, args):
        self.args = args
        self.train_step = 0
        self.gamma = self.args.gamma
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.memory = MultiAgentReplayBuffer(critic_dims, actor_dims, n_actions, n_agents, self.args)

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, self.args))

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        for agent_idx, agent in enumerate(self.agents):
            if os.path.exists(self.model_path + '/agent_%d' % agent_idx + '/actor_params.pt'):
                self.load_checkpoint(agent_idx, agent)

    def choose_action(self, obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(obs[agent_idx])
            actions.append(action)

        return actions

    def learn(self):
        if not self.memory.ready():
            return

        for agent_idx, agent in enumerate(self.agents):
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
                target_next_actions = T.cat([a_c.actor_target.forward(T.as_tensor(actor_next_states[idx], dtype=T.float32, device=self.args.device)) for idx, a_c  in enumerate(self.agents)], dim=1)
                target_next_value = agent.critic_target.forward(batch_next_states, target_next_actions).view(-1)
                target = batch_rewards[:,agent_idx] + self.gamma * (1 - batch_dones[:,agent_idx]) * target_next_value
            q_value = agent.critic.forward(batch_states, curr_actions).view(-1)
            critic_loss = (target - q_value).pow(2).mean()

            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()


            new_action = agent.actor.forward(T.as_tensor(actor_states[agent_idx], dtype=T.float32, device=self.args.device))
            new_actions_list = copy.deepcopy(curr_actions_list)
            new_actions_list[agent_idx] = new_action

            new_actions = T.cat([acts for acts in new_actions_list], dim=1)

            actor_loss = agent.critic.forward(batch_states, new_actions)
            actor_loss = -T.mean(actor_loss)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            agent._soft_update_target_network()

            self.save_checkpoint(agent_idx, agent, self.train_step)

        self.train_step += 1


    def save_checkpoint(self, agent_idx, agent, train_step):
        if train_step > 0 and train_step % self.args.save_rate == 0:
            print('... saving agent_{} ...'.format(agent_idx))
            num = str(train_step // self.args.save_rate)
            model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
            if not os.path.exists(model_path):
                    os.mkdir(model_path)
            model_path = os.path.join(model_path, 'agent_%d' % agent_idx)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            T.save(agent.actor.state_dict(), model_path + '/' + num + '_actor_params.pt')
            T.save(agent.critic.state_dict(), model_path + '/' + num + '_critic_params.pt')

    def load_checkpoint(self, agent_idx, agent):
        agent.actor.load_state_dict(T.load(self.model_path + '/agent_%d' % agent_idx + '/actor_params.pt'))
        agent.critic.load_state_dict(T.load(self.model_path +'/agent_%d' % agent_idx + '/critic_params.pt'))
        print('Agent {} successfully loaded actor: {}'.format(agent_idx, self.model_path + '/agent_%d' % agent_idx + '/actor_params.pt'))
        print('Agent {} successfully loaded critic: {}'.format(agent_idx, self.model_path + '/agent_%d' % agent_idx + '/critic_params.pt'))
