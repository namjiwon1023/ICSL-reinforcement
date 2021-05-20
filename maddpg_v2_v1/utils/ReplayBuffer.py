import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, critic_dims, actor_dims, n_actions, n_agents, args):
        self.args = args
        self.max_size = self.args.buffer_size

        self.use_cuda = self.args.use_cuda
        self.device = self.args.device

        self.ptr, self.cur_len = 0, 0
        self.count = 0

        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.n_actions = n_actions

        self.actor_state_buffer = []
        self.actor_next_state_buffer = []
        self.actor_action_buffer = []

        if self.use_cuda:
            self.state_buffer = T.empty((self.max_size, critic_dims), dtype=T.float32, device=self.device)
            self.next_state_buffer = T.empty((self.max_size, critic_dims), dtype=T.float32, device=self.device)
            self.reward_buffer = T.empty((self.max_size, n_agents), dtype=T.float32, device=self.device)
            self.done_buffer = T.empty((self.max_size,n_agents), dtype=bool, device=self.device)

            for i in range(self.n_agents):
                self.actor_state_buffer.append(T.empty((self.max_size, self.actor_dims[i]), dtype=T.float32, device=self.device))
                self.actor_next_state_buffer.append(T.empty((self.max_size, self.actor_dims[i]), dtype=T.float32, device=self.device))
                self.actor_action_buffer.append(T.empty((self.max_size, self.n_actions), dtype=T.float32, device=self.device))

        else:
            self.state_buffer = np.empty((self.max_size, critic_dims), dtype=np.float32)
            self.next_state_buffer = np.empty((self.max_size, critic_dims), dtype=np.float32)
            self.reward_buffer = np.empty((self.max_size, n_agents), dtype=np.float32)
            self.done_buffer = np.empty((self.max_size,n_agents), dtype=bool)

            for i in range(self.n_agents):
                self.actor_state_buffer.append(np.empty((self.max_size, self.actor_dims[i]), dtype=np.float32))
                self.actor_next_state_buffer.append(np.empty((self.max_size, self.actor_dims[i]), dtype=np.float32))
                self.actor_action_buffer.append(np.empty((self.max_size, self.n_actions), dtype=np.float32))

    def store_transition(self, obs, state, action, reward, next_obs, next_state, done):
        if self.use_cuda:
            for agent_idx in range(self.n_agents):
                self.actor_state_buffer[agent_idx][self.ptr] = T.as_tensor(obs[agent_idx], dtype=T.float32, device=self.device)
                self.actor_next_state_buffer[agent_idx][self.ptr] = T.as_tensor(next_obs[agent_idx], dtype=T.float32, device=self.device)
                self.actor_action_buffer[agent_idx][self.ptr] = T.as_tensor(action[agent_idx], dtype=T.float32, device=self.device)

            self.state_buffer[self.ptr] = T.as_tensor(state, dtype=T.float32, device=self.device)
            self.next_state_buffer[self.ptr] = T.as_tensor(next_state, dtype=T.float32, device=self.device)
            self.reward_buffer[self.ptr] = T.as_tensor(reward, dtype=T.float32, device=self.device)
            self.done_buffer[self.ptr] = T.as_tensor(done, dtype=bool, device=self.device)

        else:
            for agent_idx in range(self.n_agents):
                self.actor_state_buffer[agent_idx][self.ptr] = obs[agent_idx]
                self.actor_next_state_buffer[agent_idx][self.ptr] = next_obs[agent_idx]
                self.actor_action_buffer[agent_idx][self.ptr] = action[agent_idx]

            self.state_buffer[self.ptr] = state
            self.next_state_buffer[self.ptr] = next_state
            self.reward_buffer[self.ptr] = reward
            self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)
        self.count += 1

    def sample_buffer(self, batch_size=None):
        if batch_size == None:
            batch_size = self.args.batch_size

        if self.args.fast_start:
            if self.count < batch_size:
                index = np.random.choice(self.cur_len, self.count)
            else:
                index = np.random.choice(self.cur_len, batch_size, replace = False)
        else:
            index = np.random.choice(self.cur_len, batch_size, replace = False)

        actor_states = []
        actor_next_states = []
        actions = []

        if self.use_cuda:
            states = T.as_tensor(self.state_buffer[index], dtype=T.float32, device=self.device)
            rewards = T.as_tensor(self.reward_buffer[index], dtype=T.float32, device=self.device)
            next_states = T.as_tensor(self.next_state_buffer[index], dtype=T.float32, device=self.device)
            dones = T.as_tensor(self.done_buffer[index], dtype=bool, device=self.device)

            for agent_idx in range(self.n_agents):
                actor_states.append(T.as_tensor(self.actor_state_buffer[agent_idx][index], dtype=T.float32, device=self.device))
                actor_next_states.append(T.as_tensor(self.actor_next_state_buffer[agent_idx][index], dtype=T.float32, device=self.device))
                actions.append(T.as_tensor(self.actor_action_buffer[agent_idx][index], dtype=T.float32, device=self.device))

        else:
            states = self.state_buffer[index]
            rewards = self.reward_buffer[index]
            next_states = self.next_state_buffer[index]
            dones = self.done_buffer[index]

            for agent_idx in range(self.n_agents):
                actor_states.append(self.actor_state_buffer[agent_idx][index])
                actor_next_states.append(self.actor_next_state_buffer[agent_idx][index])
                actions.append(self.actor_action_buffer[agent_idx][index])

        return actor_states, states, actions, rewards, actor_next_states, next_states, dones

    def ready(self):
        if self.cur_len >= self.args.batch_size:
            return True

    def __len__(self):
        return self.cur_len