import numpy as np
import matplotlib.pyplot as plt


def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    args.n_players = env.n
    args.n_agents = env.n - args.num_adversaries
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]
    args.high_action = 1
    args.low_action = -1
    return env, args


def _Static_plot(scores, figure_file):
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.plot(scores, "r-", linewidth=1.5, label="Episode Reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc="best", shadow=True)
    plt.title('Return')
    plt.savefig(figure_file)