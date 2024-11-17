import numpy as np

def debug(args):
    # extract args
    L = args.n_ode_group
    Q = args.n_arm

    # load model
    path = '../data/model/ts_5.0_0.npz'
    model_data = np.load(path)

    # load trace
    path = '../data/simulator_trace/5_5.0.npz'
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
