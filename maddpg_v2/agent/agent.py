import torch as T
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
import copy

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, args):
        self.args = args
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.device = self.args.device

        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx

        self.actor = ActorNetwork(actor_dims, n_actions, name=self.agent_name+'_actor', self.args)
        self.critic = CriticNetwork(critic_dims, n_agents, n_actions, name=self.agent_name+'_critic', self.args)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for q in self.critic_target.parameters():
            q.requires_grad = False

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def choose_action(self, observation):
        with T.no_grad():
            state = T.as_tensor([observation], dtype=T.float, device=self.device)
            actions = self.actor(state)
            noise = T.rand(self.n_actions, device=self.device)
            action = actions + noise

        return action.detach().cpu().numpy()[0]

    def _soft_update_target_network(self):
        tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

            for t_p, l_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)


    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

