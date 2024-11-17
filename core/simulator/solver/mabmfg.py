import torch
import time

from .base import BaseSolver
from simulator import noma

class MabMfgSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    ################################
    # LIST OF METHODS FOR ALL SOLVER
    ################################
    def p_tx_constraint(self, p_tx):
        # extract args
        args       = self.args
        p_tx_max   = args.p_tx_max
        n_violated = 0
        violated   = False
        # pass
        n_violated = len(torch.where(p_tx > p_tx_max)[0])
        if n_violated > 0:
            violated = True
        return violated, n_violated

    def compute_reward(self):
        # extract args
        args           = self.args
        idx            = self.active_indices
        n_active       = len(idx)
        Q              = args.n_group
        temporary_g_db = self.temporary_g_db
        arms           = self.arms
        mean_g_db      = torch.zeros(Q, dtype=torch.float)
        rewards        = self.rewards
        gamma_opt      = self.gamma_opt
        # reset rewards to 0
        rewards[idx] = 0
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
                rewards[idx[idx1]]   = torch.clamp(1 - p_tx / 0.0025,
                                                   min=0, max=1)
                self.p_tx[idx[idx1]] = p_tx
                # compute constraint
                idx2 = noma.compute_sinr_constraint(p_tx, h, gamma_opt, args)
                # rewards[idx[idx1[idx2]]] = 0
                n_constraint_violated += len(idx2)
        # self.constraint_violated = n_constraint_violated / n_active if n_active > 0 else 0
        self.constraint_violated = False
        self.n_sinr_violated = n_constraint_violated
        if n_constraint_violated > 0:
            self.constraint_violated = True

    def extract_info(self):
        # extract args
        args                = self.args
        Q                   = args.n_arm
        active_indices      = self.active_indices
        rewards             = self.rewards
        n_active            = len(self.active_indices)
        population_profile  = self.population_profile
        constraint_violated = self.constraint_violated
        p_tx                = self.p_tx
        # check sinr constraint
        n_sinr_violated           = self.n_sinr_violated
        # check max power constraint
        violated, n_p_tx_violated = self.p_tx_constraint(p_tx)
        # filter reward of active devices only
        info = {
            'avg_reward'         : torch.mean(rewards[active_indices]).item(),
            'max_p_tx'           : torch.max(p_tx[active_indices]).item(),
            'avg_p_tx'           : torch.mean(p_tx[active_indices]).item(),
            'n_p_tx_violated'    : n_p_tx_violated,
            'n_sinr_violated'    : n_sinr_violated,
            'n_violated'         : n_p_tx_violated + n_sinr_violated,
            'n_active'           : n_active,
        }
        for q in range(Q):
            info[f'arm_{q}'] = 0
        return info

    ##########################
    # LOGIC FOR EACH TIME SLOT
    ##########################
    def step(self):
        tic = time.time()
        self.active_indices = self.traffic_model.get_active_device()
        self.sample_g_db()
        self.choose_arm()
        self.compute_reward()
        self.update_state()
        toc                = time.time()
        info               = self.extract_info()
        info['time']       = toc - tic
        info['step']       = self.global_step
        self.add_info(info)
        self.monitor.step(info)
        self.cache_trace()
        self.global_step += 1

    ##############################
    # LIST OF METHODS TO CUSTOMIZE
    ##############################
    def initialize_state(self):
        # extract args
        args = self.args
        K_tot     = len(self.traffic_model.x_d)
        # initialize arm (selected group, not power level)
        self.arms = torch.zeros(K_tot, dtype=torch.int)
        self.gamma_opt = noma.compute_gamma_opt()
        self.d = torch.sqrt(self.traffic_model.x_d ** 2 + \
                            self.traffic_model.y_d ** 2)

    def choose_arm(self):
        # extract args
        args     = self.args
        Q        = args.n_group
        idx      = self.active_indices
        n_active = len(idx)
        # first, just choose a random group
        self.arms[idx] = torch.randint(low=0, high=Q, size=(n_active, ),
                                       dtype=torch.int)

    def update_state(self):
        # extract parameter
        args = self.args

    def add_info(self, info):
        args      = self.args
        idx       = self.active_indices
        exp_coeff = args.exp_coeff
        Q         = args.n_arm
        nu        = args.exploration_rate
        info['min_var'] = 0
