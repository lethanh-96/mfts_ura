import numpy as np
import torch
import os

from .base import BaseSolver

class ThompsonSamplingSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract parameter
        args  = self.args
        Q     = args.n_arm
        K_tot = len(self.traffic_model.x_d)
        # initialize state
        self.mean = torch.ones([K_tot, Q])
        self.var  = torch.ones([K_tot, Q])
        self.c    = torch.ones([K_tot, Q])
        # initialize trace
        self.cache_mu   = []
        self.cache_c    = []
        self.trace_mu   = []
        self.trace_c    = []

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        active_indices = self.active_indices
        n_active       = len(active_indices)
        active_var     = self.var[active_indices, :]
        active_mean    = self.mean[active_indices, :]
        arms           = self.arms
        # extract parameters
        theta         = torch.randn(n_active, Q) * torch.sqrt(active_var) + active_mean
        selected_arms = torch.argmax(theta, dim=1).to(dtype=torch.int)
        # assign the selected arms
        arms[active_indices] = selected_arms

    def update_state(self):
        # extract parameter
        idx      = self.active_indices
        rewards  = self.rewards
        mean     = self.mean
        args     = self.args
        arms     = self.arms
        var      = self.var
        c        = self.c
        args     = self.args
        exp_coeff= args.exp_coeff
        r_min    = args.r_min
        r_max    = args.r_max
        d        = self.traffic_model.d[idx]
        Q        = args.n_arm
        n_active = len(idx)
        idx_violated = self.idx_violated
        n_violated   = len(idx_violated)
        # bayesian update
        arms_idx             = arms[idx]
        mean[idx, arms_idx] += (rewards[idx] - mean[idx, arms_idx]) / c[idx, arms_idx] ** exp_coeff
        c[idx, arms_idx]    += 1
        var[idx, arms_idx]   = 1 / c[idx, arms_idx] ** exp_coeff
        # partially reset half of the violated player
        if n_violated > 0:
            n_reset   = int(n_violated * 0.5)
            idx_reset = idx_violated[torch.randperm(n_violated)[:n_reset]]
            mean[idx_reset] = 1
            c[idx_reset]    = 1
            var[idx_reset]  = 1

    def save_model(self):
        args = self.args
        label = f'{args.solver}_{args.expected_n_event:0.1f}_{args.seed}'
        path = os.path.join(args.model_dir, f'{label}.npz')
        kwargs = {
            'g_db': self.g_db.detach().cpu().numpy(),
            'mean': self.mean.detach().cpu().numpy(),
            'c'   : self.c.detach().cpu().numpy(),
        }
        np.savez_compressed(path, **kwargs)

    def cache_trace(self):
        # extract args
        args = self.args
        # add to cache
        self.cache_mu.append(self.mean.detach().cpu().numpy())
        self.cache_c.append(self.c.detach().cpu().numpy())
        # add to trace if cache full
        if len(self.cache_mu) >= int(args.n_step / args.n_approximator_sample):
            self.cache_mu = np.array(self.cache_mu)
            self.cache_c = np.array(self.cache_c)
            self.trace_mu.append(np.mean(self.cache_mu, axis=0))
            self.trace_c.append(np.mean(self.cache_c, axis=0))
            self.cache_mu   = []
            self.cache_c    = []

    def save_trace(self):
        # extract args
        args = self.args
        # create saving path
        path = os.path.join(args.simulator_trace_dir,
                            f'{args.n_arm}_{args.expected_n_event:0.1f}.npz')
        # save trace
        np.savez_compressed(path, g_db=self.g_db.detach().cpu().numpy(),
                                  mu=np.array(self.trace_mu),
                                  c=np.array(self.trace_c))
