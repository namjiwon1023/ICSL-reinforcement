import numpy as np
from agent.maddpg import MADDPG
from utils.make_env import make_env
from utils.utils import obs_list_to_state_vector, _random_seed, _Static_plot
from utils.arguments import get_args
import matplotlib.pyplot as plt
import os

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
    _random_seed(args.seed)

    env = make_env(args.scenario_name)

    n_agents = env.n

    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    critic_dims = sum(actor_dims)
    n_actions = env.action_space[0].n

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, args)

    model_path = args.save_dir + '/' + args.scenario_name
    if os.path.exists(model_path):
        maddpg_agents.load_checkpoint()
    else:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    total_steps = 0
    score_history = []
    best_score = 0

    if not args.evaluate:
        print('-----------learning start--------------')
        for i in range(1, args.total_episodes + 1):
            obs = env.reset()
            state = obs_list_to_state_vector(obs)
            score = 0
            done = [False]*n_agents
            episode_step = 0

            np.savetxt("./Total_scores.txt", score_history, delimiter=",")

            while not any(done):

                actions = maddpg_agents.choose_action(obs)

                obs_, reward, done, info = env.step(actions)
                next_state = obs_list_to_state_vector(obs_)

                if episode_step >= args.max_episode_len:
                    done = [True]*n_agents

                maddpg_agents.memory.store_transition(obs, state, actions, reward, obs_, next_state, done)

                if total_steps % 100 == 0 and not args.evaluate:
                    maddpg_agents.learn()

                obs = obs_

                score += sum(reward)
                total_steps += 1
                episode_step += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if not args.evaluate:
                if avg_score > best_score:
                    maddpg_agents.save_checkpoint()
                    best_score = avg_score

            if i % args.print_iter == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))
                _Static_plot(score_history, args.save_dir + '/' + args.scenario_name)

            if total_steps >= args.time_steps:
                print('Exceed the total number of training steps !')
                print('Over !')
                break
    else:
        print('-----------evaluate start--------------')
        evaluate(env, args, maddpg_agents)