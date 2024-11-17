from simulator import traffic, uma, monitor
import torch
import time
import tqdm

class BaseSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize traffic model
        self.traffic_model = traffic.create_traffic_model(args)
        # initialize monitor
        self.monitor = monitor.Monitor(args)

    ################################
    # LIST OF METHODS FOR ALL SOLVER
    ################################
    def base_initialize_state(self):
        # extract args
        args  = self.args
        K_tot = len(self.traffic_model.x_d)
        Q     = args.n_arm
        n_p   = args.n_pilot_bit
        m     = args.bit_rate
        N_0   = args.noise_power
        # initialize state
        self.population_profile  = torch.full((Q, ), 1 / Q, dtype=torch.float)
        self.active_indices      = None
        self.rewards             = torch.zeros(K_tot, dtype=torch.float)
        self.p_tx                = torch.zeros(K_tot, dtype=torch.float)
        self.arms                = torch.zeros(K_tot, dtype=torch.int)
        self.g_db                = uma.compute_g_db(self.traffic_model.x_d, self.traffic_model.y_d, args)
        self.temporary_g_db      = torch.zeros(K_tot, dtype=torch.float)
        self.pi                  = self.compute_receiving_power()
        self.sinr_min_db         = uma.get_sinr_min(n=n_p, m=m, N_0=N_0)
        self.constraint_violated = False
        self.n_violated          = 0
        self.idx_violated        = None
        self.global_step         = 0

    def compute_population_profile(self):
        # extract args
        args = self.args
        Q    = args.n_arm
        arms = self.arms
        active_indices = self.active_indices
        # count the arms distribution of active devices
        active_arms = arms[active_indices]
        for q in range(Q):
            self.population_profile[q] = torch.sum(active_arms == q)
        # normalize the distribution
        if torch.sum(self.population_profile) == 0:
            self.population_profile[:] = 1
        self.population_profile /= torch.sum(self.population_profile)

    def sample_g_db(self):
        # extract args
        args  = self.args
        K_tot = len(self.traffic_model.x_d)
        sigma2shadow   = args.sigma2shadow
        active_indices = self.active_indices
        n_active       = len(active_indices)
        temporary_g_db = self.temporary_g_db
        g_db           = self.g_db
        # sample zeta and add to g_db
        zeta = torch.randn(n_active)
        temporary_g_db[active_indices] = g_db[active_indices] + zeta * sigma2shadow

    def compute_receiving_power(self):
        # extract args
        args  = self.args
        Q     = args.n_arm
        N_0   = args.noise_power
        # compute receiving power, compare to N0
        if Q == 1:
            pi_N0_db = torch.tensor([-3])
        else:
            pi_N0_db = torch.linspace(1, 0, Q) * (args.rx_power_max - args.rx_power_min) + args.rx_power_min
        pi = 10 ** (pi_N0_db / 10) * N_0
        return pi

    def compute_transmission_power(self):
        # extract args
        args           = self.args
        active_indices = self.active_indices
        Q              = args.n_arm
        arms           = self.arms
        active_arms    = arms[active_indices]
        p_tx           = self.p_tx
        temporary_g_db = self.temporary_g_db
        pi             = self.pi
        # compute p_tx parallely for each arm
        for q in range(Q):
            active_q_indices = active_indices[torch.where(active_arms == q)]
            active_q_g_db    = temporary_g_db[active_q_indices]
            active_q_g       = 10 ** (active_q_g_db / 10)
            p_tx[active_q_indices] = uma.get_p_tx(pi[q], active_q_g)

    def sinr_constraint(self, sinr_db):
        # extract args
        args           = self.args
        sinr_min_db    = self.sinr_min_db
        Q              = args.n_arm
        arms           = self.arms
        active_indices = self.active_indices
        active_arms    = arms[active_indices]
        n_violated     = 0
        violated       = False
        # check every arm that have active device
        idx_violated = []
        for q in range(Q):
            active_q_indices = active_indices[torch.where(active_arms == q)]
            if len(active_q_indices):
                if sinr_db[q] < sinr_min_db:
                    n_violated += len(active_q_indices)
                    idx_violated.append(active_q_indices)
                    violated = True
        if len(idx_violated) > 0:
            idx_violated = torch.cat(idx_violated)
        else:
            idx_violated = torch.tensor([])
        return violated, n_violated, idx_violated

    def p_tx_constraint(self, p_tx):
        # extract args
        args       = self.args
        p_tx_max   = args.p_tx_max
        n_violated = 0
        violated   = False
        #
        idx_violated = torch.where(p_tx > p_tx_max)[0]
        n_violated   = len(idx_violated)
        if n_violated > 0:
            violated = True
        return violated, n_violated, idx_violated

    def compute_reward(self):
        # extract args
        args               = self.args
        K_tot              = len(self.traffic_model.x_d)
        population_profile = self.population_profile
        arms               = self.arms
        active_indices     = self.active_indices
        rewards            = self.rewards
        sinr_min_db        = self.sinr_min_db
        n_active           = len(active_indices)
        M                  = args.n_antenna
        N_0                = args.noise_power
        n_d                = args.n_data_bit
        n_p                = args.n_pilot_bit
        m                  = args.bit_rate
        pi                 = self.pi
        p_tx               = self.p_tx
        # reset rewards to 0
        rewards[active_indices] = 0
        # compute transmission power
        self.compute_transmission_power()
        # compute sinr average form
        self.avg_sinr_db = uma.get_average_group_sic_sinr(pi, population_profile, n_active,
                                                          M=M, N_0=N_0, n_d=n_d, n_p=n_p, m=m)
        # compute sinr sampled form
        self.sinr_db = uma.get_group_sic_sinr(pi, population_profile, n_active,
                                              M=M, N_0=N_0, n_d=n_d, n_p=n_p, m=m)
        # check sinr constraint
        violated1, n_violated1, idx_violated1 = self.sinr_constraint(self.avg_sinr_db)
        self.constraint_violated = violated1
        # check max power constraint
        violated2, n_violated2, idx_violated2 = self.p_tx_constraint(p_tx)
        # save number of violated
        self.n_violated = n_violated1 + n_violated2
        self.idx_violated = torch.unique(torch.cat([idx_violated1, idx_violated2])).to(dtype=torch.int)
        # now reward is power only
        rewards[active_indices] = torch.clamp(1 - p_tx[active_indices] / 0.0025,
                                              min=0, max=1)
        return violated1, violated2

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
        sinr_db             = self.sinr_db
        avg_sinr_db         = self.avg_sinr_db
        # check sinr constraint
        violated, n_sinr_violated, _ = self.sinr_constraint(self.avg_sinr_db)
        self.constraint_violated = violated
        # check max power constraint
        violated, n_p_tx_violated, _ = self.p_tx_constraint(p_tx)
        # filter reward of active devices only
        if len(active_indices) == 0:
            info = {
                'avg_reward'      : 0,
                'avg_p_tx'        : 0,
                'max_p_tx'        : 0,
                'n_p_tx_violated' : 0,
                'n_sinr_violated' : 0,
                'n_violated'      : 0,
                'n_active'        : 0,
            }
        else:
            info = {
                'avg_reward'      : torch.mean(rewards[active_indices]).item(),
                'avg_p_tx'        : torch.mean(p_tx[active_indices]).item(),
                'max_p_tx'        : torch.max(p_tx[active_indices]).item(),
                'n_p_tx_violated' : n_p_tx_violated,
                'n_sinr_violated' : n_sinr_violated,
                'n_violated'      : n_p_tx_violated + n_sinr_violated,
                'n_active'        : n_active,
            }
        for q in range(Q):
            info[f'arm_{q}'] = population_profile[q].item()
        return info

    ##########################
    # LOGIC FOR EACH TIME SLOT
    ##########################
    def step(self):
        args = self.args
        tic = time.time()
        self.active_indices = self.traffic_model.get_active_device()
        self.sample_g_db()
        self.choose_arm()
        self.compute_population_profile()
        self.compute_reward()
        self.update_state()
        toc                = time.time()
        info               = self.extract_info()
        info['time']       = toc - tic
        info['step']       = self.global_step
        self.add_info(info)
        self.monitor.step(info)
        if args.save_trace:
            self.cache_trace()
        self.global_step += 1

    ###############################
    # ONLY NEED TO CALL THIS TO RUN
    ###############################
    def simulate(self):
        args = self.args
        tic = time.time()
        self.base_initialize_state()
        self.initialize_state()
        try:
            for t in range(self.args.n_step):
                self.step()
                toc = time.time()
                if args.solver == 'centralized':
                    if toc - tic > args.max_simulation_time:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if args.save_model:
                self.save_model()
            if args.save_trace:
                self.save_trace()
            if args.csv:
                self.monitor.write_csv()

    ##############################
    # LIST OF METHODS TO CUSTOMIZE
    ##############################
    def initialize_state(self):
        pass

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        active_indices = self.active_indices
        # select arms randomly
        self.arms[active_indices] = torch.randint(low=0, high=Q, size=(len(active_indices), ), dtype=torch.int)

    def update_state(self):
        pass

    def add_info(self, info):
        args      = self.args
        idx       = self.active_indices
        exp_coeff = args.exp_coeff
        Q         = args.n_arm
        nu        = args.exploration_rate
        if 'ts' in args.solver:
            c = self.c
            info['min_var'] = torch.mean(torch.min(1 / c[idx] ** exp_coeff, dim=1)[0]).item()
        elif args.solver == 'hedge':
            info['min_var'] = nu / Q
        elif args.solver == 'random':
            info['min_var'] = 1 / Q
        elif args.solver == 'centralized':
            info['min_var'] = 1
        elif args.solver == 'greedy':
            info['min_var'] = 0
        elif args.solver == 'weighted_random':
            info['min_var'] = 1 / Q
        elif args.solver == 'mfdqn':
            pass
        else:
            raise NotImplementedError

    def save_model(self):
        pass

    def cache_trace(self):
        pass

    def save_trace(self):
        pass
    #############################################################################
