import torch

from .base import BaseSolver

class HedgeSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract parameter
        args  = self.args
        Q     = args.n_arm
        K_tot = len(self.traffic_model.x_d)
        # initialize state
        self.state = torch.ones([K_tot, Q])
        self.t     = torch.zeros([K_tot])

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        active_indices = self.active_indices
        n_active       = len(active_indices)
        active_state   = self.state[active_indices, :]
        active_t       = self.t[active_indices]
        arms           = self.arms
        nu             = args.exploration_rate
        beta           = args.smoothing_coeff
        # construct arm selection distribution
        sigma = (1 - nu) * torch.exp(beta * active_state) / torch.sum(torch.exp(beta * active_state), dim=1).unsqueeze(-1) + nu / Q
        # cumulative distribution for faster random choice in parallel
        sigma_cumsum = torch.cumsum(sigma, dim=1)
        # random choice in parallel
        r = torch.rand(n_active)
        selected_arms = torch.zeros(n_active, dtype=torch.int)
        for q in range(Q - 1):
            idx = torch.where(r > sigma_cumsum[:, q])[0]
            selected_arms[idx] = q + 1
        # assign the selected arms
        self.arms[active_indices] = selected_arms

    def update_state(self):
        # extract parameter
        idx      = self.active_indices
        rewards  = self.rewards
        args     = self.args
        arms     = self.arms
        state    = self.state
        t        = self.t
        gamma    = 1 / (t[idx] + 1)
        arms_idx = arms[idx]
        r_min    = args.r_min
        r_max    = args.r_max
        d        = self.traffic_model.d[idx]
        Q        = args.n_arm
        # iterative update of Hedge policy
        state[idx, arms_idx] = (1 - gamma) * state[idx, arms_idx] + gamma * rewards[idx]
        if not self.constraint_violated:
            state[idx, arms_idx] = (1 - gamma) * state[idx, arms_idx] + gamma * rewards[idx]
        # if constraint violated update mean with probability depend on distance
        else:
            state[idx, arms_idx] = (1 - gamma) * state[idx, arms_idx] + gamma * 0
#             p_apply_constraint_1                  = (d - r_min) / (r_max - r_min)
#             p_apply_constraint_2                  = (Q - arms[idx]) / Q
#             p_apply_constraint                    = p_apply_constraint_1 * p_apply_constraint_2
#             r                                     = torch.rand_like(p_apply_constraint)
#             idx_apply_constraint                  = torch.where(r < p_apply_constraint)[0]
#             personal_reward                       = rewards[idx]
#             personal_reward[idx_apply_constraint] = 0
#             state[idx, arms_idx] = (1 - gamma) * state[idx, arms_idx] + gamma * personal_reward
        t[idx]              += 1
        # print(state[idx, :][0, :])
