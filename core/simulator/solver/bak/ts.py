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
        # bayesian update
        arms_idx             = arms[idx]
        # if not violate constraint, update mean and var normally
        if not self.constraint_violated:
            mean[idx, arms_idx] += (rewards[idx] - mean[idx, arms_idx]) / c[idx, arms_idx] ** exp_coeff
            c[idx, arms_idx]    += 1
        # if constraint violated update mean with probability depend on distance
        else:
            # partially update mean
            p_apply_constraint_1       = (d - r_min) / (r_max - r_min)
            p_apply_constraint_2       = (Q - arms[idx]) / Q
            p_apply_constraint         = p_apply_constraint_1 * p_apply_constraint_2
            r                          = torch.rand_like(p_apply_constraint)
            idx_apply_constraint       = torch.where(r < p_apply_constraint)[0]
            diff                       = (rewards[idx] - mean[idx, arms_idx])
            diff[idx_apply_constraint] = 0
            mean[idx, arms_idx]       += diff / c[idx, arms_idx] ** exp_coeff
            # partially update var
            c[idx, arms_idx]          += 1
            # c[idx[idx_apply_constraint], arms_idx[idx_apply_constraint]] -= 1
        # update var
        var[idx, arms_idx]   = 1 / c[idx, arms_idx] ** exp_coeff
        # find idx of all device that all second best arms is eliminated
        mean = self.mean[idx]
        c    = self.c[idx]
        var  = self.var[idx]
        top_k_values, top_k_indices = torch.topk(mean, k=2, dim=-1)
        best_arm_mean = top_k_values[:, 0]
        best_arm_idx  = top_k_indices[:, 0]
        best_arm_std  = torch.sqrt(1 / c[:, best_arm_idx] ** exp_coeff)
        second_arm_mean = top_k_values[:, 1]
        second_arm_idx  = top_k_indices[:, 1]
        second_arm_std  = torch.sqrt(1 / c[:, second_arm_idx] ** exp_coeff)
        mask = ((best_arm_mean - best_arm_std) - (second_arm_mean + second_arm_std)) > 0
        # reset
        idx_reset = torch.where(mask)[0]
        mean[idx_reset] = 1
        c[idx_reset]    = 1
        var[idx_reset]  = 1
        # reset 10% of the amount of user that has contraint violated
        n_reset   = int(self.n_violated / n_active)
        idx_reset = idx[torch.randperm(n_active)[:n_reset]]
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
