import numpy as np
from agent.maddpg import MADDPG
from utils.make_env import make_env
from utils.utils import obs_list_to_state_vector, _random_seed, _Static_plot
from utils.arguments import get_args
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import copy

def evaluate(env, agents, args):
    returns = []
    for episode in range(args.evaluate_episodes):
        s = env.reset()
        rewards = 0
        for time_step in range(args.evaluate_episode_len):
            env.render()
            actions = agents.choose_action(s)
            u = copy.deepcopy(actions)
            u.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = env.step(u)
            rewards += r[0]
            s = s_next
        returns.append(rewards)
        print('Returns is', rewards)
    return sum(returns) / args.evaluate_episodes

if __name__ == '__main__':
    args = get_args()
    _random_seed(args.seed)

    # env parameters
    env = make_env(args.scenario_name, args.benchmark)
    n_agents = env.n - 1
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = env.action_space[0].n

    #setting
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, args)
    model_path = args.save_dir + '/' + args.scenario_name

    total_steps = 0
    # score_history = []

    if not args.evaluate:
        print('-----------------------------------------------------')
        print('-----------------learning start----------------------')
        print('-----------------------------------------------------')
        returns = []
        for i in tqdm(range(1, args.total_episodes + 1)):
            obs = env.reset()
            state = obs_list_to_state_vector(obs[:n_agents])
            score = 0
            episode_step = 0
            np.savetxt("./return.txt", returns, delimiter=",")

            while True:
                actions = maddpg_agents.choose_action(obs)
                u = copy.deepcopy(actions)
                u.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                obs_, reward, done, info = env.step(u)
                next_state = obs_list_to_state_vector(obs_[:n_agents])

                maddpg_agents.memory.store_transition(obs[:n_agents], state, actions, reward[:n_agents], obs_[:n_agents], next_state, done[:n_agents])
                all_done = all(done)
                terminal = (episode_step >= args.max_episode_len)
                obs = obs_

                maddpg_agents.learn()

                # score += sum(reward)
                total_steps += 1
                episode_step += 1

                if total_steps % args.evaluate_rate == 0 :
                    returns.append(evaluate(env, maddpg_agents, args))
                    plt.figure()
                    plt.plot(range(len(returns)), returns)
                    plt.xlabel('episode * ' + str(args.evaluate_rate / args.max_episode_len))
                    plt.ylabel('average returns')
                    plt.savefig(model_path + '/plt.png', format='png')

                if all_done or terminal:
                    break

            # score_history.append(score)
            # avg_score = np.mean(score_history[-100:])

            # if i % args.print_iter == 0:
            #     print('Episode : {} | Step : {} | Return : {} | Mean : {} |'.foramt(i, total_steps, score, avg_score))

            if total_steps >= args.time_steps:
                print('-------------------------------------------')
                print('Exceed the total number of training steps !')
                print('Over !')
                print('-------------------------------------------')
                # _Static_plot(score_history, args.save_dir + '/' + args.scenario_name)
                break
    else:
        print('-----------------------------------------------------')
        print('-----------------evaluate start----------------------')
        print('-----------------------------------------------------')
        returns = evaluate(env, maddpg_agents, args)
        print('Average returns is', returns)
