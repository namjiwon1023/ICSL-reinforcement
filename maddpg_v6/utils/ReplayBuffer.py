import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, critic_dims, actor_dims, n_actions, n_agents, args):
        self.args = args
        self.max_size = self.args.buffer_size

        self.ptr, self.cur_len = 0, 0
        self.count = 0

        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.n_actions = n_actions

        self.actor_state_buffer = []
        self.actor_next_state_buffer = []
        self.actor_action_buffer = []

        self.state_buffer = np.empty((self.max_size, critic_dims), dtype=np.float32)
        self.next_state_buffer = np.empty((self.max_size, critic_dims), dtype=np.float32)
        self.reward_buffer = np.empty((self.max_size, n_agents), dtype=np.float32)
        self.done_buffer = np.empty((self.max_size,n_agents), dtype=np.float32)

        for i in range(self.n_agents):
            self.actor_state_buffer.append(np.empty((self.max_size, self.actor_dims[i]), dtype=np.float32))
            self.actor_next_state_buffer.append(np.empty((self.max_size, self.actor_dims[i]), dtype=np.float32))
            self.actor_action_buffer.append(np.empty((self.max_size, self.n_actions), dtype=np.float32))

    def store_transition(self, obs, state, action, reward, next_obs, next_state, done):
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

        index = np.random.choice(self.cur_len, batch_size, replace = False)

        actor_states = []
        actor_next_states = []
        actions = []

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