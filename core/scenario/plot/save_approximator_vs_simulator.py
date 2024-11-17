import matplotlib.pyplot as plt
import numpy as np
import os

def save_approximator_vs_simulator(args):
    # extract args
    L = args.n_ode_group
    Q = args.n_arm
    # load approximator
    path = os.path.join(args.approximator_trace_dir,
                        f'{L}_{Q}_{args.expected_n_event:0.1f}.npz')
    data   = np.load(path)
    mu_a   = data['mu'].reshape(-1, L, Q)
    c_a    = data['c'].reshape(-1, L, Q)
    g_db_a     = data['g_db']
    g_db_a_min = data['g_db_min']
    g_db_a_max = data['g_db_max']
    # load simulator
    path = os.path.join(args.simulator_trace_dir,
                        f'{Q}_{args.expected_n_event:0.1f}.npz')
    data   = np.load(path)
    mu_s   = data['mu']
    c_s    = data['c']
    g_db_s = data['g_db']
    # extract parameter
    K_tot = len(g_db_s)
    # group user
    group_idx = np.zeros(K_tot)
    for k in range(K_tot):
        group_idx[k] = np.argmin((g_db_s[k] - g_db_a)**2)
    # for each group
    mu_sl   = np.zeros([L, Q])
    c_sl    = np.zeros([L, Q])
    for l in range(L):
        # plot approximator mean
        for q in range(Q):
            plt.plot(mu_a[:, l, q], label=f'approximator-{q}')
            # plot simulator mean
            # for q in range(Q):
            idx = np.where(group_idx == l)[0]
            mu_s_l = np.mean(mu_s[:, idx, q], axis=1)
            plt.plot(mu_s_l, label=f'simulator-{q}')
            # transfer value
            value = np.max(mu_a[-100:, l, q])
            plt.axhline(value, color='k', linestyle='--', label='transfer-value')
            # decorate
            plt.xlabel('time')
            plt.ylabel('$\mu$')
            plt.legend()
            plt.ylim((0, 1))
            # save figure
#             path = os.path.join(args.figure_dir, f'{args.scenario}_{args.expected_n_event}_mu_{l}.pdf')
#             plt.tight_layout()
#             plt.savefig(path)
#             plt.clf()

            # plot approximator count
            # for q in range(Q):
            plt.plot(c_a[:, l, q], label=f'approximator-{q}')
            # plot simulator mean
            # for q in range(Q):
            idx = np.where(group_idx == l)[0]
            c_s_l = np.mean(c_s[:, idx, q], axis=1)
            plt.plot(c_s_l, label=f'simulator-{q}')
            # transfer value
            value = np.max(c_a[:, l, q])
            plt.axhline(value, color='k', linestyle='--', label='transfer-value')
            # decorate
            plt.xlabel('time')
            plt.ylabel('$c$')
            plt.legend()
            # save figure
#             path = os.path.join(args.figure_dir, f'{args.scenario}_{args.expected_n_event}_c_{l}.pdf')
#             plt.tight_layout()
#             plt.savefig(path)
#             plt.clf()

            # store reference value
            mu_sl[l, q] = mu_s_l[-1]
            c_sl[l, q]  = c_s_l[-1]

    # save reference value
    path = os.path.join(args.approximator_trace_dir,
                        f'ref_{L}_{Q}_{args.expected_n_event:0.1f}.npz')
    np.savez_compressed(path, g_db=g_db_a,
                              g_db_min=g_db_a_min,
                              g_db_max=g_db_a_max,
                              mu=mu_sl, c=c_sl)
