import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser("Multi Agent Deep Deterministic Policy Gradient")

    parser.add_argument("--device", default=device, help="Using GPU or CPU")
    parser.add_argument("--seed", type=int, default=3, help="random seed")

    parser.add_argument("--n-hiddens-1", type=int, default=64, help="Neural Networks hidden layers units")
    parser.add_argument("--n-hiddens-2", type=int, default=64, help="Neural Networks hidden layers units")

    parser.add_argument("--scenario-name", type=str, default="simple_adversary", help="name of the scenario script")

    parser.add_argument("--max-episode-len", type=int, default=30, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--actor-lr", type=float, default=1e-2, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-2, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise-rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()
