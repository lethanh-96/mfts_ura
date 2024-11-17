import numpy as np
import torch
import os

from ..base import BaseSolver
from .model import DQN

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract args
        args = self.args
        # initialize DQN
        self.policy_net = DQN(args)
        self.target_net = DQN(args)
        target_net.load_state_dict(policy_net.state_dict())
        # initialize optimizer
        optimizer = optim.AdamW(policy_net.parameters(), lr=args.learning_rate, amsgrad=True)
        memory = ReplayMemory(10000)

    def choose_arm(self):
        # extract args
        args       = self.args
        Q          = args.n_arm
        idx        = self.active_indices
        n_active   = len(idx)
        policy_net = self.policy_net
        # extract observation
        observation    = self.temporary_g_db
        mf_observation = self.population_profile / n_active
        # compute exploration rate
        eps = args.eps_end + (args.eps_start - eps_end) * math.exp(-1 * self.global_step / args.eps_decay)
        #
        if np.random.rand() > eps:
            self.arms[idx] = torch.randint(low=0, high=Q, size=(n_active, ))
        else:
            arms = policy_net(observation, mf_observation).max(1).indices
