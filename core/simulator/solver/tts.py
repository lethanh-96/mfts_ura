import numpy as np
import torch
import time
import os

from .ts import ThompsonSamplingSolver

class TransferThompsonSamplingSolver(ThompsonSamplingSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract parameter
        args      = self.args
        Q         = args.n_arm
        K_tot     = len(self.traffic_model.x_d)
        exp_coeff = args.exp_coeff
        # initialize state
        self.mean = torch.ones([K_tot, Q])
        self.var  = torch.ones([K_tot, Q])
        self.c    = torch.ones([K_tot, Q])
        # load from trained model
        g_db, mean, c = self.load_model()
        # transfer from trained model
        print('[+] transfering from trained model')
        tic = time.time()
        for k in range(K_tot):
            g_db_k  = self.g_db[k]
            d       = (g_db - g_db_k) ** 2
            indices = d.topk(k=3, largest=False).indices
            self.mean[k]   = torch.mean(mean[indices, :])
            self.c[k]      = torch.mean(c[indices, :])
            self.var[k, :] = 1 / self.c[k, :] ** exp_coeff
        toc = time.time()
        print(f'    - completed in: {toc - tic:0.1f}s')

    def load_model(self):
        args  = self.args
        label = f'ts_{args.expected_n_event:0.1f}_0'
        path  = os.path.join(args.model_dir, f'{label}.npz')
        data  = np.load(path)
        g_db  = torch.tensor(data['g_db'])
        mean  = torch.tensor(data['mean'])
        c     = torch.tensor(data['c'])
        return g_db, mean, c
