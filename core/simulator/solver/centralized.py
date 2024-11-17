import numpy as np
import itertools
import torch

from .base import BaseSolver

class CentralizedSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract parameter
        args  = self.args
        Q     = args.n_arm
        K_tot = len(self.traffic_model.x_d)
        # initialize state
        pass

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        idx            = self.active_indices
        n_active       = len(idx)
        g_db_active    = self.temporary_g_db[idx]
        arms           = self.arms
        rewards        = self.rewards
        step           = args.centralized_step
        # if the system is not saturated yet, don't optimize
        if self.global_step < 1000:
            return
        # initialize best solution
        best_reward = 0
        best_xs     = None
        # sort user by g_db_ascending
        sorted_idx     = idx[torch.sort(g_db_active)[1]]
        # function to check if solution xs is sorted or not
        def is_sorted(l):
            if l[-1] == n_active:return False
            return all(l[i] < l[i+1] for i in range(len(l) - 1))
        # function to decode from solution xs to selected arms
        def decode_arms(xs):
            # sequentially assign power level 0 -> Q-1
            lb = 0
            ub = 0
            for q in range(Q):
                # assign ub for this level
                ub = xs[q]
                # assign arms
                arms[sorted_idx[lb:ub]] = q
                # assign lb for next level
                lb = ub
                # print(q, len(np.where(self.arms[idx] == q)[0]))
            self.compute_population_profile()
        for xs in itertools.product(np.arange(0, n_active, step), repeat=Q-1):
            if is_sorted(xs):
                xs = np.append(xs, n_active)
                decode_arms(xs)
                violated1, violated2 = self.compute_reward()
                avg_reward = torch.mean(rewards[idx]).item()
                if violated1 or violated2:
                    pass # violated either constraint
                elif avg_reward > best_reward:
                    best_reward = avg_reward
                    best_xs     = xs
                    # print(f'[+] found xs={xs} avg_reward={avg_reward}')
        # decode the best reward again
        if best_xs is None:
            # print('[-] no feasible solution found'e
            best_xs = xs # avoid exception
        else:
            # print(f'[-] solution found: {best_xs}')
            pass
        try:
            decode_arms(best_xs)
        except:
            print('[+] Hello error')
            pass

    def update_state(self):
        pass
