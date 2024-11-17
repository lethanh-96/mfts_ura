import torch
import random
import numpy as np

from .benamor import BenamorSolver
from simulator import noma

class BenamorV2Solver(BenamorSolver):

    def __init__(self, args):
        super().__init__(args)

    ##############################
    # LIST OF METHODS TO CUSTOMIZE
    ##############################
    def initialize_state(self):
        # extract args
        args  = self.args
        K_tot = len(self.traffic_model.x_d)
        Q     = args.n_group
        # initialize arm (selected group, not power level)
        self.arms = torch.zeros(K_tot, dtype=torch.int)
        self.gamma_opt = noma.compute_gamma_opt()
        self.d = torch.sqrt(self.traffic_model.x_d ** 2 + \
                            self.traffic_model.y_d ** 2)
        # epsilon greedy state to select group instead of random
        self.state            = torch.zeros(K_tot, Q)
        self.count            = torch.ones(K_tot, Q)
        self.estimated_reward = torch.zeros(K_tot, Q)

    def choose_arm(self):
        # extract args
        args      = self.args
        Q         = args.n_group
        idx       = self.active_indices
        n_active  = len(idx)
        epsilon_0 = args.epsilon_0
        count            = self.count[idx]
        estimated_reward = self.estimated_reward[idx]
        # calculate epsilon
        epsilon = epsilon_0 / (torch.sum(count, dim=1) - Q + 1)
        # random action
        r = torch.rand(n_active)
        idx1 = torch.where(r <= epsilon)[0]
        self.arms[idx[idx1]] = torch.randint(low=0, high=Q, size=(len(idx1), ),
                                             dtype=torch.int)
        # greedy action
        idx1 = torch.where(r > epsilon)[0]
        max_values = torch.max(estimated_reward[idx1], dim=1)[0]
        max_maskes = estimated_reward[idx1] == max_values.view(-1, 1)
        p = max_maskes / torch.sum(max_maskes, dim=1).view(-1, 1)
        self.arms[idx[idx1]] = torch.multinomial(p, 1).squeeze().to(dtype=torch.int)

    def update_state(self):
        # extract args
        args           = self.args
        idx            = self.active_indices
        n_active       = len(idx)
        Q              = args.n_group
        temporary_g_db = self.temporary_g_db
        arms           = self.arms
        mean_g_db      = torch.zeros(Q, dtype=torch.float)
        gamma_opt      = self.gamma_opt
        t_mab          = args.t_mab
        rewards        = self.rewards
        # for each group
        n_constraint_violated = 0
        for q in range(Q):
            # extract selected active channel coefficient
            idx1 = torch.where(arms[idx] == q)[0]
            if len(idx1) > 0:
                h    = temporary_g_db[idx[idx1]]
                d    = self.d[idx[idx1]]
                # compute power
                p_tx = noma.compute_opt_benamor_p_tx(h, d, gamma_opt, args)
                # compute reward
                rewards[idx[idx1]]   = torch.clamp(1 - p_tx / 0.02,
                                                   min=0, max=1)
                self.p_tx[idx[idx1]] = p_tx
                # compute constraint
                idx2 = noma.compute_sinr_constraint(p_tx, h, gamma_opt, args)
                idx1 = torch.tensor(np.setdiff1d(np.arange(len(idx)),
                                       idx2.detach().cpu().numpy()))
                self.state[idx[idx1], self.arms[idx[idx1]]] += 1
                self.count[idx, self.arms[idx]] += 1
                self.estimated_reward[idx, self.arms[idx]] = \
                        self.state[idx, self.arms[idx]] / \
                        self.count[idx, self.arms[idx]]
        # reset after t_mab
        idx1 = torch.where(self.count[idx] > t_mab)[0]
        self.state[idx[idx1]] = 0
        self.count[idx[idx1]] = 0
        self.estimated_reward[idx[idx1]] = 0
