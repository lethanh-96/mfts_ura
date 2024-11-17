import numpy as np
import torch.optim as optim
import torch
import math
import os

from .replay_memory import ReplayMemory
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def choose_arm(self):
        # extract args
        args       = self.args
        Q          = args.n_arm
        idx        = self.active_indices
        n_active   = len(idx)
        policy_net = self.policy_net
        # extract observation
        observation    = self.temporary_g_db[idx]
        mf_observation = self.population_profile / n_active
        # compute exploration rate
        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1 * self.global_step / args.eps_decay)
        # select arm
        if np.random.rand() > eps:
            self.arms[idx] = torch.randint(low=0, high=Q, size=(n_active, ), dtype=torch.int)
        else:
            self.arms[idx] = policy_net(observation, mf_observation).max(1).indices.to(dtype=torch.int)
