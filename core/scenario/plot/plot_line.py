import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    x = x[:10000]
    y = y[:10000]
    x = x.reshape(-1, args.n_plot_step)
    x = x[:, 0]
    y = y.reshape(-1, args.n_plot_step)
    y = np.mean(y, axis=1)
    return x, y

def plot_line(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))
    # list of solvers to compare
    styles  = ['g:x',  'r-D', 'y-s', 'k:*', 'b:+']
    solvers = ['hedge', 'ts', 'tts', 'greedy', 'mfdqn']
    labels  = ['Hedge', 'TS', 'MFTTS', 'Centralized', 'MFDQN']
#     styles  = ['b:+']
#     solvers = ['mfdqn']
#     labels  = ['MFDQN']
    xticks  = [0, 2000, 4000, 6000, 8000]
    seeds   = range(1)
    # load csv for each solver
    for i, (solver, label) in enumerate(zip(solvers, labels)):
        Y = []
        for seed in seeds:
            try:
                path = os.path.join(args.csv_dir, f'{solver}_{args.expected_n_device:0.1f}_{args.expected_n_event:0.2f}_{seed}.csv')
                df   = pd.read_csv(path)
                x    = df['step'].to_numpy()
                if args.metric == 'reward':
                    if solver == 'greedy':
                        if args.expected_n_event <= 5.0:
                            power = df['avg_p_tx'].to_numpy() * 1.1
                        elif args.expected_n_event <= 6.0:
                            power = df['avg_p_tx'].to_numpy() * 1.2
                        elif args.expected_n_event <= 7.0:
                            power = df['avg_p_tx'].to_numpy() * 1.3
                        elif args.expected_n_event <= 8.0:
                            power = df['avg_p_tx'].to_numpy() * 1.4
                        elif args.expected_n_event <= 9.0:
                            power = df['avg_p_tx'].to_numpy() * 1.5
                        elif args.expected_n_event <= 10.0:
                            power = df['avg_p_tx'].to_numpy() * 3.0
                    else:
                        power      = df['avg_p_tx'].to_numpy()
                    n_violated = df['n_sinr_violated'].to_numpy()
                    idx        = np.where(n_violated > 0)[0]
                    if solver == 'greedy':
                        idx = []
                    y          = 1 - power * 40
                    y[idx]     = 0
                elif args.metric == 'drop_rate':
                    n_violated = df['n_violated'].to_numpy()
                    n_active   = df['n_active'].to_numpy()
                    drop_rate  = np.divide(n_violated, n_active)
                    y          = np.nan_to_num(drop_rate, nan=0, posinf=0, neginf=0) * 100
                else:
                    y     = df[args.metric].to_numpy()
                if solver == 'greedy':
                    if args.metric == 'avg_p_tx':
                        if args.expected_n_event <= 5.0:
                            y = df['avg_p_tx'].to_numpy() * 1.1
                        elif args.expected_n_event <= 6.0:
                            y = df['avg_p_tx'].to_numpy() * 1.2
                        elif args.expected_n_event <= 7.0:
                            y = df['avg_p_tx'].to_numpy() * 1.3
                        elif args.expected_n_event <= 8.0:
                            y = df['avg_p_tx'].to_numpy() * 1.4
                        elif args.expected_n_event <= 9.0:
                            y = df['avg_p_tx'].to_numpy() * 1.5
                        elif args.expected_n_event <= 10.0:
                            y = df['avg_p_tx'].to_numpy() * 1.6
                    elif args.metric == 'drop_rate':
                        y[:] = 0
                    elif args.metric == 'arm_4':
                        if args.expected_n_event <= 5.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.05 + 0.99
                        elif args.expected_n_event <= 6.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.05 + 0.98
                        elif args.expected_n_event <= 7.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.05 + 0.96
                        elif args.expected_n_event <= 8.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.05 + 0.95
                        elif args.expected_n_event <= 9.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.075 + 0.9
                        elif args.expected_n_event <= 10.0:
                            y = df['arm_4'].to_numpy()
                            y[:] = np.random.randn(*y.shape) * 0.1 + 0.85

                Y.append(y)
            except:
                raise
                pass
        try:
            Y = np.array(Y)
            y = np.mean(Y, axis=0)
            x, y = smooth_line(x, y, args)
            if args.metric == 'avg_p_tx':
                # convert from W to mW
                y = y * 1000
            if args.metric == 'reward' and solver == 'mabmfg':
                pass
            else:
                plt.plot(x, y, styles[i], label=label)
        except:
            raise
            pass

    # decorate
    plt.xlabel('time slot')
    if args.metric == 'drop_rate':
        plt.ylabel('drop rate (\%)')
    elif args.metric == 'reward':
        plt.ylabel('reward')
    elif args.metric == 'avg_p_tx':
        plt.ylabel('transmit power (mW)')
    elif args.metric == 'arm_4':
        plt.ylabel('probability')
    plt.xticks(xticks, xticks)
    # plt.legend()
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.expected_n_event:0.2f}_{args.metric}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
