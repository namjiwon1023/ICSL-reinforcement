import numpy as np
from make_env import make_env

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

env = make_env('simple_adversary')
print('number of agents', env.n)
print('observation space', env.observation_space)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()
print(observation)

no_op = np.array([1.12, 0.12, 0.343, 0.12342, 0.435])
# action = [no_op, no_op, no_op, no_op]
action = [no_op, no_op, no_op]
obs_, reward, done, info = env.step(action)
print(reward)
print(done)

# # state = np.array([])
# # for obs in observation:
# #     state = np.append(state, obs)
# state = obs_list_to_state_vector(observation)
# print(state.shape)
'''
env : simple_tag
number of agents 4
observation space [Box(16,), Box(16,), Box(16,), Box(14,)]
action space [Discrete(5), Discrete(5), Discrete(5), Discrete(5)]
n actions 5
reward : [0, 0, 0, -0.19955622346591428]
done : [False, False, False, False]
'''

'''
env : simple_adversary
number of agents 3
observation space [Box(8,), Box(10,), Box(10,)]
action space [Discrete(5), Discrete(5), Discrete(5)]
n actions 5
reward : [-1.1759351933071736, 0.35322387743578265, 0.35322387743578265]
done : [False, False, False]
'''