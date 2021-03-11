# Deep Reinforcement Learning Baselines (Model Free)(Under Construction)

## Value Based

### Online / Offline

- [ ] Deep Q Network(DQN) (off-policy)
- [ ] Double Deep Q Network(Double DQN) (off-policy)
- [ ] Dueling Deep Q Network(Dueling DQN) (off-policy)
- [ ] Duelling Double Deep Q Network(D3QN) (off-policy)

----------------------------------------------------------------

## Actor-Critic Method

### Online

- [ ] Advantage Actor-Critic(A2C) (on-policy)
- [ ] Asynchronous Advantage Actor-Critic(A3C) (on-policy)
- [ ] Proximal Policy Optimization(PPO)(GAE) (on-policy)(Nearing off-policy)

### Online / Offline

- [ ] Proximal Policy Gradient(PPG) (on-policy PPO + off-policy Critic[Let it share parameters with PPO's Critic])
- [ ] Deep Deterministic Policy Gradient(DDPG) (off-policy)
- [ ] Twin Delayed Deep Deterministic policy gradient(TD3) (off-policy)
- [ ] Soft Actor-Critic(SAC) (off-policy)

## Imitation Learning / Inverse Reinforcement Learning

- [ ] Behavior Cloning
- [ ] Generative Adversarial Imitation Learning

## ReplayBuffer Structure

- [ ] Prioritized Experience Replay
- [ ] Hindsight Experience Replay


## Required Python Libraries

```
pip install torch torchvision  
pip install numpy  
pip install gym  
pip install matplotlib  
pip install segment-tree  
pip install gym
```

## Papers
01. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518(7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
02. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
03. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
04. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
05. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
06. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv preprint arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
07. [R. S. Sutton, "Learning to predict by the methods of temporal differences." Machine learning, 3(1):9–44, 1988.](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
08. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv preprint arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)
09. [M. Babaeizadeh et al., "Reinforcement learning through asynchronous advantage actor-critic on a gpu.", International Conference on Learning Representations, 2017.](https://arxiv.org/pdf/1611.06256)
10. [J. Schulman et al., "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/abs/1707.06347.pdf)
11. [T. P. Lillicrap et al., "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/pdf/1509.02971.pdf)
12. [S. Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods." arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
13. [T.  Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv preprint arXiv:1801.01290, 2018.](https://arxiv.org/pdf/1801.01290.pdf)
14. [M. Vecerik et al., "Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards."arXiv preprint arXiv:1707.08817, 2017](https://arxiv.org/pdf/1707.08817.pdf)
15. [A. Nair et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations." arXiv preprint arXiv:1709.10089, 2017.](https://arxiv.org/pdf/1709.10089.pdf)