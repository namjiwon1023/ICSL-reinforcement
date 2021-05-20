from utils.arguments import get_args
from utils.utils import make_env, _Static_plot
from algorithm.maddpg import MADDPG
import numpy as np
import random
import torch as T
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate(env, agents, args):
    returns = []
    for episode in range(args.evaluate_episodes):
        s = env.reset()
        rewards = 0
        for time_step in range(args.evaluate_episode_len):
            env.render()
            actions = agents.choose_action(s)
            s_next, r, done, info = env.step(actions)
            rewards += r[0]
            s = s_next
        returns.append(rewards)
        print('Returns is', rewards)
    return sum(returns) / args.evaluate_episodes

if __name__ == '__main__':
    args = get_args()
    env, args = make_env(args)
    maddpg_agents = MADDPG(args)
    noise = args.noise_rate
    epsilon = args.epsilon
    best_score = -np.inf
    episode = 0
    model_path = args.save_dir + '/' + args.scenario_name
    if os.path.exists(model_path):
        # maddpg_agents.load_checkpoint()
        pass
    else:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    if args.evaluate:
        returns = evaluate(env, maddpg_agents, args)
        print('Average returns is', returns)
    else:
        returns = []
        score = 0
        score_history = []
        for time_step in tqdm(range(args.time_steps)):
            if time_step % args.max_episode_len == 0:
                episode += 1
                s = env.reset()
                score_history.append(score)
                np.savetxt("./Total_scores.txt", score_history, delimiter=",")
                avg_score = np.mean(score_history[-10:])
                if not args.evaluate:
                    if avg_score > best_score:
                        maddpg_agents.save_checkpoint()
                        best_score = avg_score
                score = 0
            with T.no_grad():
                actions = maddpg_agents.choose_action(s, noise, epsilon)
            s_next, r, done, info = env.step(actions)
            maddpg_agents.memory.store_transition(s[:args.n_agents], actions, r[:args.n_agents], s_next[:args.n_agents])
            s = s_next

            maddpg_agents.learn()
            score += sum(r)
            if time_step > 0 and time_step % 100 == 0:
                print('episode', episode, 'time step', time_step, 'average score {:.1f}'.format(avg_score))
            # if time_step > 0 and time_step % args.evaluate_rate == 0:
            #     returns.append(evaluate())
            #     plt.figure()
            #     plt.plot(range(len(returns)), returns)
            #     plt.xlabel('episode * ' + str(args.evaluate_rate / episode_limit))
            #     plt.ylabel('average returns')
            #     plt.savefig(save_path + '/plt.png', format='png')
            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, noise - 0.0000005)

        _Static_plot(score_history, args.save_dir + '/' + args.scenario_name)


