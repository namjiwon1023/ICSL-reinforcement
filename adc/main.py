import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.ReplayBuffer import ReplayBuffer
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork

from algorithm.ADC import ADCAgent
from utils.utils import _random_seed

if __name__ == '__main__':
    writer = SummaryWriter()
    _random_seed(123)
    params = {
                'gamma' : 0.9,
                'actor_lr' : 1e-3,
                'critic_lr' : 2e-3,
                'tau' : 1e-2,
                'memory_size' : 5000,
                'batch_size' : 32,
                'total_step' : 0,
                'render' : False,
                'eval_steps' : 1000,
                'gradient_steps' : 1000,
                'target_update_interval' : 1,
                'learning_steps' : 0,
                'epsilon' : 0.1,
            }

    agent = ADCAgent(**params)

    i_episode = int(1e6)
    max_steps = int(3e6)

    best_score = agent.env.reward_range[0]

    scores = []
    store_scores = []
    eval_rewards = []

    avg_score = 0
    n_updates = 0

    for i in range(1, i_episode + 1):
        state = agent.env.reset()
        cur_episode_steps = 0
        score = 0
        done = False

        while not done:

            if agent.render is True:
                agent.env.render()

            cur_episode_steps += 1
            agent.total_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            real_done = False if cur_episode_steps >= agent.env.spec.max_episode_steps else done
            mask = 0.0 if real_done else agent.gamma
            agent.transition += [reward, next_state, mask]
            agent.memory.store(*agent.transition)
            state = next_state
            score += reward

            if agent.total_step % agent.gradient_steps == 0 and agent.memory.ready():
                value_losses,  Policy_losses = agent.learn()
                n_updates += 1

            if agent.total_step % agent.eval_steps == 0 and agent.memory.ready():
                running_reward = np.mean(scores[-10:])
                eval_reward = agent.evaluate_agent(n_starts=10)
                eval_rewards.append(eval_reward)
                writer.add_scalar('Loss/Value', value_losses, n_updates)
                writer.add_scalar('Loss/Policy', Policy_losses, n_updates)
                writer.add_scalar('Reward/Train', running_reward, agent.total_step)
                writer.add_scalar('Reward/Test', eval_reward, agent.total_step)
                print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                scores = []

        scores.append(score)
        store_scores.append(score)
        avg_score = np.mean(store_scores[-10:])

        np.savetxt("./Episode_return.txt", store_scores, delimiter=",")
        np.savetxt("./Step_return.txt", eval_rewards, delimiter=",")

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if agent.total_step >= max_steps:
            print('Reach the maximum number of training steps ！')
            break

        print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} | Learning Step : {} | update number : {} |'.format(i, round(score, 2), round(avg_score, 2), agent.total_step, agent.learning_steps, n_updates))

    agent.env.close()