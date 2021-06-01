import numpy as np
import torch as T

class ReplayBuffer:
    def __init__(self, memory_size, n_states, batch_size):

        self.batch_size = batch_size

        self.state = np.empty([memory_size, n_states], dtype=np.float32)
        self.next_state = np.empty([memory_size, n_states], dtype=np.float32)
        # self.action = np.empty([memory_size, n_actions],dtype=np.float32)
        self.action = np.empty([memory_size],dtype=np.float32)
        self.reward = np.empty([memory_size], dtype=np.float32)
        self.mask = np.empty([memory_size],dtype=np.float32)

        self.max_size = memory_size
        self.ptr, self.cur_len, = 0, 0
        self.n_states = n_states
        self.count = 0

    def store(self, state, action, reward, next_state, mask):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.mask[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)
        self.count += 1

    def sample_batch(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        index = np.random.choice(self.cur_len, batch_size, replace = False)

        return dict(state = self.state[index],
                    action = self.action[index],
                    reward = self.reward[index],
                    next_state = self.next_state[index],
                    mask = self.mask[index],
                    )

    def ready(self):
        if self.cur_len >= self.batch_size:
            return True

    def __len__(self):
        return self.cur_len