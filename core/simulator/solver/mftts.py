import numpy as np
import torch
import time
import os

from .ts import ThompsonSamplingSolver

class MeanFieldTransferThompsonSamplingSolver(ThompsonSamplingSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract parameter
        args  = self.args
        Q     = args.n_arm
        K_tot = len(self.traffic_model.x_d)
        exp_coeff = args.exp_coeff
        # initialize state
        self.mean = torch.ones([K_tot, Q])
        self.var  = torch.ones([K_tot, Q])
        self.c    = torch.ones([K_tot, Q])
        # load from trained model
        g_db, mean, c = self.load_model()
        # transfer from approximated model
        print('[+] transfering from trained model')
        tic = time.time()
        for k in range(K_tot):
            g_db_k  = self.g_db[k]
            d       = (g_db - g_db_k) ** 2
            indices = d.topk(k=1, largest=False).indices
            _ = torch.max(mean[indices, :], dim=0)
            self.mean[k, :] = mean[indices, :]
            self.c[k, :]    = c[indices, :]
            self.var[k, :] = 1 / self.c[k, :] ** exp_coeff
        toc = time.time()
        print(f'    - completed in: {toc - tic:0.1f}s')

    def load_model(self):
        # extract args
        args = self.args
        Q = args.n_arm
        L = args.n_ode_group
        # load trace
        path = os.path.join(args.simulator_trace_dir,
                            f'{Q}_{args.expected_n_event:0.1f}.npz')
        simulator_data = np.load(path)
        # extract last step data from simulator trace
        g_db   = simulator_data['g_db']
        mu     = simulator_data['mu'][-1, ...]
        c      = simulator_data['c'][-1, ...]
        K_tot  = len(g_db)
        # sort data
        idx = np.argsort(g_db)
        g_db = g_db[idx]
        mu   = mu[idx]
        c    = c[idx]
        # building g_db group
        g_db_a = []
        for item in g_db[:int(len(g_db) // L * L)].reshape(L, -1):
            g_db_a.append(np.mean(item))
        g_db_a = np.array(g_db_a)
        # grouping user by g_db
        group_idx = np.zeros(K_tot)
        for k in range(K_tot):
            group_idx[k] = np.argmin((g_db[k] - g_db_a)**2)
        # extract parameter by group
        mu_a = []
        c_a = []
        for l in range(L):
            idx = np.where(group_idx == l)[0]
            mu_a.append(np.mean(mu[idx, :], axis=0))
            c_a.append(np.mean(c[idx, :], axis=0))
        mu_a = np.array(mu_a)
        c_a  = np.array(c_a)
        g_db_a = torch.tensor(g_db_a)
        mu_a = torch.tensor(mu_a)
        c_a = torch.tensor(c_a)
        return g_db_a, mu_a, c_a

#     def load_model(self):
#         # extract args
#         args  = self.args
#         L    = args.n_ode_group
#         Q    = args.n_arm
#         # load trace
# #         path = os.path.join(args.approximator_trace_dir,
# #                             f'{L}_{Q}_{args.expected_n_event:0.1f}.npz')
# #         data  = np.load(path)
# #         # extract data of the last time step
# #         g_db  = torch.tensor(data['g_db'])
# #         mean  = torch.tensor(data['mu']).reshape(-1, L, Q)
# #         c     = torch.tensor(data['c']).reshape(-1, L, Q)
#         path = os.path.join(args.approximator_trace_dir,
#                             f'ref_{L}_{Q}_{args.expected_n_event:0.1f}.npz')
#         data  = np.load(path)
#         # extract data of the last time step
#         g_db  = torch.tensor(data['g_db'])
#         mean  = torch.tensor(data['mu'])
#         c     = torch.tensor(data['c'])
#         return g_db, mean, c
