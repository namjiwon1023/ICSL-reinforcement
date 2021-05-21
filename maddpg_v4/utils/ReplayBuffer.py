import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.max_size = args.buffer_size

        self.args = args

        self.ptr, self.cur_len = 0, 0

        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['observations_%d' % i] = np.empty([self.max_size, self.args.obs_shape[i]])
            self.buffer['actions_%d' % i] = np.empty([self.max_size, self.args.action_shape[i]])
            self.buffer['rewards_%d' % i] = np.empty([self.max_size])
            self.buffer['next_observations_%d' % i] = np.empty([self.max_size, self.args.obs_shape[i]])

        self.lock = threading.Lock()

    def store_transition(self, observation, action, reward, next_observation):
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['observations_%d' % i][self.ptr] = observation[i]
                self.buffer['actions_%d' % i][self.ptr] = action[i]
                self.buffer['rewards_%d' % i][self.ptr] = reward[i]
                self.buffer['next_observations_%d' % i][self.ptr] = next_observation[i]

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)


    def sample_batch(self, batch_size=None):
        if batch_size == None:
            batch_size = self.args.batch_size

        batch_buffer = {}
        index = np.random.choice(self.cur_len, batch_size, replace = False)
        for key in self.buffer.keys():
            batch_buffer[key] = self.buffer[key][index]

        return batch_buffer

    def ready(self):
        if self.cur_len >= self.args.batch_size:
            return True

    def __len__(self):
        return self.cur_len