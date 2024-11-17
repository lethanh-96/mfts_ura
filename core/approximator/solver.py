from simulator import uma as simulator_uma
from . import uma as approximator_uma
from simulator import traffic

import scipy.integrate as itg
import numpy as np
import sympy as sp
import torch
import time
import os

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize traffic model
        self.traffic_model = traffic.create_traffic_model(args)
        # compute receiving power
        self.pi = self.compute_receiving_power()
        # extract g group
        self.extract_g_group()

    def extract_g_group(self):
        # extract args
        args = self.args
        L    = args.n_ode_group
        # initialize state
        g_db = simulator_uma.compute_g_db(self.traffic_model.x_d, self.traffic_model.y_d, args)
        g_db = g_db.detach().cpu().numpy()
        # sort the g_db
        g_db = np.array(sorted(g_db))
        # initialize
        self.g_db_min  = []
        self.g_db_max  = []
        self.g_db_mean = []
        for item in g_db[:int(len(g_db) // L * L)].reshape(L, -1):
            self.g_db_min.append(item.min())
            self.g_db_max.append(item.max())
            self.g_db_mean.append(np.mean(item))
        self.g_db = self.g_db_mean

    def compute_receiving_power(self):
        # extract args
        args  = self.args
        Q     = args.n_arm
        N_0   = args.noise_power
        # compute receiving power, compare to N0
        pi_N0_db = torch.linspace(1, 0, Q) * (args.rx_power_max - args.rx_power_min) + args.rx_power_min
        pi = 10 ** (pi_N0_db / 10) * N_0
        return pi.detach().cpu().numpy()

    #####################################

    def create_symbols(self):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        # create symbols
        y = sp.symbols(f'y:{2*L*Q}')
        t = sp.symbols('t')
        return y, t

    #####################################

    def get_p_l(self, l, y):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm

        # compute arm's selection probability per partition l
        p_l = []
        for q in range(Q):
            # compute probability of q being an selected arm
            p_lq = 1
            for j in range(Q):
                if q != j:
                    # compute probability of arm q being selected arm over arm j
                    mu1   = y[l * Q + q]
                    mu2   = y[l * Q + j]
                    mu12  = mu1 - mu2
                    var1  = 1 / (y[L * Q + l * Q + q] + 1)
                    var2  = 1 / (y[L * Q + l * Q + j] + 1)
                    var12 = var1 + var2
                    p_lq  = p_lq * 0.5 * sp.erfc(-(mu12) / (np.sqrt(2) * sp.sqrt(var12)))

            p_l.append(p_lq)
        return p_l

    def get_P(self, y):
        # extract args
        args = self.args
        L    = args.n_ode_group
        #
        P = [self.get_p_l(l, y) for l in range(L)]
        return P

    def create_population_profile_expr(self, y):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        #
        P = self.get_P(y)
        population_profile = []
        for q in range(Q):
            f_q = sum(P[l][q] for l in range(L))
            population_profile.append(f_q)
        population_profile = np.array(population_profile)
        return population_profile

    #####################################

    def create_sinr_db_expr(self, population_profile, K_a):
        args = self.args
        sinr_db = approximator_uma.get_group_sic_sinr(pi=self.pi,
                                                      xi=population_profile,
                                                      Ka=K_a,
                                                      M=args.n_antenna,
                                                      N_0=args.noise_power,
                                                      n_d=args.n_data_bit,
                                                      n_p=args.n_pilot_bit,
                                                      m=args.bit_rate)
        return sinr_db

    def r(self, l, q, sinr_db): # reward function
        # extract parameter
        pi = self.pi
        g_db = self.g_db
        # compute transmission power
        g_l = 10 ** (g_db[l] / 10)
        r = sp.Max(1 - pi[q] * g_l / 0.0025, 0)
        return r

    #####################################

    def get_P(self, y):
        # extract args
        args = self.args
        L    = args.n_ode_group
        #
        P = [self.get_p_l(l, y) for l in range(L)]
        return P

    def create_dc_dt_expr(self, y, p_a):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        #
        P = self.get_P(y)
        derivatives = []
        for l in range(L):
            for q in range(Q):
                d = p_a * P[l][q]
                derivatives.append(d)
        return derivatives

    def create_dmu_dt_expr(self, y, p_a, sinr_db):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        #
        P = self.get_P(y)
        derivatives = []
        for l in range(L):
            for q in range(Q):
                d = p_a * P[l][q] / (y[L * Q + l * Q + q]) * (self.r(l, q, sinr_db) - y[l * Q + q])
                derivatives.append(d)
        return derivatives

    #####################################

    def get_initial_condition(self):
        # extract args
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        #
        y0 = np.ones([2 * L * Q])
        return y0

    def get_t_eval(self):
        # extract parameters
        args = self.args
        T    = args.n_step
        t_eval = np.linspace(0, T-1, args.n_approximator_sample)
        return t_eval, T

    #####################################

    def save_trace(self):
        # extract parameters
        args = self.args
        L    = args.n_ode_group
        Q    = args.n_arm
        # create path
        path = os.path.join(args.approximator_trace_dir,
                            f'{L}_{Q}_{args.expected_n_event:0.1f}.npz')
        # extract data
        mu   = self.y[ : L * Q]
        c    = self.y[L * Q : 2 * L * Q]
        t    = self.t
        y    = self.y
        # save trace
        np.savez_compressed(path, g_db=self.g_db,
                                  g_db_min=self.g_db_min,
                                  g_db_max=self.g_db_max,
                                  mu=mu, c=c, t=t, y=y)

    #####################################

    def run(self, p_a, K_a):
        # record start time
        tic = time.time()
        # create symbols
        y, t = self.create_symbols()
        # create expression for population profile
        population_profile = self.create_population_profile_expr(y)
        # create expression for sinr_db
        sinr_db = self.create_sinr_db_expr(population_profile, K_a)
        # create expression for derivatives of states
        dmu_dt  = self.create_dmu_dt_expr(y, p_a, sinr_db)
        dc_dt   = self.create_dc_dt_expr(y, p_a)
        dy_dt   = dmu_dt + dc_dt
        print('[+] create ode complete in:', time.time() - tic)
        # create problem
        f = sp.lambdify((t, y), dy_dt)
        print('[+] lambdify ode complete in:', time.time() - tic)
        # get evaluation points
        t_eval, T = self.get_t_eval()
        # set initial condition
        y0 = self.get_initial_condition()
        # solve
        solution = itg.solve_ivp(f, (0, T), y0, t_eval=t_eval, rtol=1e-6)#, atol=1e-1)
        print('[+] solve ode complete in:', time.time() - tic)
        # extracting solution
        self.y = solution.y
        self.t = solution.t

        if solution.status == -1:
            print('[+] Integration step failed.')
        elif solution.status == 0:
            print('[+] The solver successfully reached the end of tspan.')
            # save trace
            self.save_trace()
        elif solution.status == 1:
            print('[+] A termination event occurred.')
        print(solution.message)
