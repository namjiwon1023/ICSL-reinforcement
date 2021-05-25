import torch as T
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
import copy

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, args):
        self.args = args
        self.tau = self.args.tau
        self.device = self.args.device

        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx

        self.actor = ActorNetwork(actor_dims, n_actions, self.agent_name + '_actor', self.args)
        self.critic = CriticNetwork(critic_dims, n_agents, n_actions, self.agent_name + '_critic', self.args)

        self.actor_target = ActorNetwork(actor_dims, n_actions, self.agent_name + '_actor_target', self.args)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = CriticNetwork(critic_dims, n_agents, n_actions, self.agent_name + '_critic_target', self.args)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        for q in self.critic_target.parameters():
            q.requires_grad = False

    def choose_action(self, observation):
        with T.no_grad():
            state = T.as_tensor([observation], dtype=T.float32, device=self.device)
            pi = self.actor(state)

        return pi.detach().cpu().numpy()[0]

    def _soft_update_target_network(self, tau=None):
        if tau == None:
            tau = self.args.tau

        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

            for t_p, l_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

