import numpy as np
from agent.maddpg import MADDPG
from ReplayBuffer import MultiAgentReplayBuffer
from make_env import make_env
from utils.utils import obs_list_to_state_vector, _random_seed
from utils.arguments import get_args

if __name__ == '__main__':
    args = get_args()
    _random_seed(args)
    env = make_env(args.scenario_name)
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    n_actions = env.action_space[0].n

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, args)

    memory = MultiAgentReplayBuffer(critic_dims, actor_dims, n_actions, n_agents, args)

    total_steps = 0
    score_history = []
    best_score = 0

    if args.evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(args.total_episodes):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if args.evaluate:
                env.render()

            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            next_state = obs_list_to_state_vector(obs_)

            if episode_step >= args.max_episode_len:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, next_state, done)

            if total_steps % 100 == 0 and not args.evaluate:
                maddpg_agents.learn(memory)

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

        if total_steps >= args.time_steps:
            print('Exceed the total number of training steps !')
            print('Over !')
            break