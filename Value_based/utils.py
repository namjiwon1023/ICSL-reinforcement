import matplotlib.pyplot as plt
import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree
import random
import os
import gym


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.show()

def plot_dynamic(scores):
    # Need to define before entering the loop : plt.ion(), plt.figure()
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for e in range(len(running_avg)):
        running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
    plt.cla()
    plt.title("Reward")
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total_Reward")
    plt.plot(scores, "r-", linewidth=1.5, label="Episode_Reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="Avg_Reward")
    plt.legend(loc="best", shadow=True)
    plt.pause(0.1)
    plt.savefig(figure_file)
    plt.show()


class ReplayBufferBasis:
    def __init__(self, memory_size, n_states, batch_size=32):
        self.state = np.zeros([memory_size, n_states], dtype=np.float32)
        self.next_state = np.zeros([memory_size, n_states], dtype=np.float32)
        self.actions = np.zeros([memory_size],dtype=np.float32)
        self.rewards = np.zeros([memory_size], dtype=np.float32)
        self.done = np.zeros([memory_size], dtype=np.float32)

        self.max_size, self.batch_size = memory_size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        index = np.random.choice(self.size, self.batch_size, replace = False)

        return dict(state = self.state[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_state[index],
                    done = self.done[index])

    def __len__(self):
        return self.size


class ReplayBufferActDim:
    def __init__(self, memory_size, n_states, n_actions, batch_size=32):
        self.state = np.zeros([memory_size, n_states], dtype=np.float32)
        self.next_state = np.zeros([memory_size, n_states], dtype=np.float32)
        self.actions = np.zeros([memory_size, n_actions],dtype=np.float32)
        self.rewards = np.zeros([memory_size], dtype=np.float32)
        self.done = np.zeros([memory_size], dtype=np.float32)

        self.max_size, self.batch_size = memory_size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        index = np.random.choice(self.size, self.batch_size, replace = False)

        return dict(state = self.state[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_state[index],
                    done = self.done[index])

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBufferBasis):
    def __init__(self, n_states, memory_size, batch_size=32, alpha=0.6):
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(n_states, memory_size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, done):
        """ Store experience and priority. """
        super().store(state, action, reward, next_state, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta=0.4):
        """ Sample a batch of experiences. """
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        state = self.state[indices]
        next_state = self.next_state[indices]
        action = self.actions[indices]
        reward = self.rewards[indices]
        done = self.done[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(state = state,
                    next_state = next_state,
                    action = action,
                    reward = reward,
                    done = done,
                    weights = weights,
                    indices = indices,
                    )

    def update_priorities(self, indices, priorities):
        """ Update priorities of sampled transitions. """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """ Sample indices based on proportions. """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        """ Calculate the weight of the experience at idx. """
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


def _layer_norm(layer, std=1.0, bias_const=1e-6):
    ''' Deep Reinforcement learning Neural network initialization
            NN.apply(_layer_norm)
    '''
    if type(layer) == nn.Linear:
        T.nn.init.orthogonal_(layer.weight, std)
        T.nn.init.constant_(layer.bias, bias_const)


class ActionNormalizer(gym.ActionWrapper):
    '''OpenAI Gym
    Map the continuous action range of the environment to [-1,1]
    env = ActionNormalizer(env)
    '''
    def action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)
        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)
        return action


class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """
    def __init__(
        self,
        action_dim,
        min_sigma = 1.0,
        max_sigma = 1.0,
        decay_period = 1000000,
    ):
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t = 0) -> float:

        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)