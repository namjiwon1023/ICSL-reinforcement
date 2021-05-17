
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import random

def _layer_norm(layer, std=1.0, bias_const=1e-6):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)


def random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)

def _plot(scores, eval_rewards):
    plt.subplot(121)
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for e in range(len(running_avg)):
        running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
    plt.cla()
    plt.title("Return")
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(scores, "r-", linewidth=1.5, label="Episode Reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="Avg Reward")
    plt.legend(loc="best", shadow=True)

    plt.subplot(122)
    plt.cla()
    plt.title("Return")
    plt.grid(True)
    plt.xlabel("Step (Unit 1000)")
    plt.ylabel("Total Reward")
    plt.plot(eval_rewards, "b-", linewidth=1.5, label="Step Reward")
    plt.legend(loc="best", shadow=True)
    plt.pause(0.1)

    plt.savefig('./sac.jpg')
    plt.show()


# Code courtesy of JPH: https://github.com/jparkerholder
def make_gif(policy, env, step_count, state_filter, maxsteps=1000):
    envname = env.spec.id
    gif_name = '_'.join([envname, str(step_count)])
    state = env.reset()
    done = False
    steps = []
    rewards = []
    t = 0
    while (not done) & (t< maxsteps):
        s = env.render('rgb_array')
        steps.append(s)
        action = policy.get_action(state, state_filter=state_filter, deterministic=True)
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        action = action.reshape(len(action), )
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    clip.write_gif('gifs/{}.gif'.format(gif_name), fps=30)