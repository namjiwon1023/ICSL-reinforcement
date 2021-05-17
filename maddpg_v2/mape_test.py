import numpy as np
from make_env import make_env

def obs_list_to_state_vector(observation):
    state = np.array([])
    print('fake state : ',state)
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

env = make_env('simple_tag')
print('number of agents', env.n)
print('observation space', env.observation_space)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()
# print(observation)

no_op = np.array([1.12, 0.12, 0.343, 0.12342, 0.435])
action = [no_op, no_op, no_op, no_op]
obs_, reward, done, info = env.step(action)
print(reward)
print(done)

'''
env : simple_tag
number of agents 4
observation space [Box(16,), Box(16,), Box(16,), Box(14,)]
action space [Discrete(5), Discrete(5), Discrete(5), Discrete(5)]
n actions 5
reward : [0, 0, 0, -0.19955622346591428]
done : [False, False, False, False]
'''