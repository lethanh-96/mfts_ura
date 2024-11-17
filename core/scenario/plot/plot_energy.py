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

def plot_energy(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))
    # list of solvers to compare
    styles  = ['g:x',  'r-D', 'y-s', 'm-h', 'k:*']
    solvers = ['hedge', 'ts', 'tts', 'mftts', 'greedy']
    labels  = ['Hedge', 'TS', 'MFTTS', 'TTS', 'Centralized']
    xticks  = [0, 2000, 4000, 6000, 8000]
    seeds   = range(10)
    energies = []
    # load csv for each solver
    for i, (solver, label) in enumerate(zip(solvers, labels)):
        E = []
        for seed in seeds:
            try:
                path = os.path.join(args.csv_dir, f'{solver}_{args.expected_n_device:0.1f}_{args.expected_n_event:0.2f}_{seed}.csv')
                df   = pd.read_csv(path)
                x    = df['step'].to_numpy()
                Ka   = df['n_active']
                p    = df['avg_p_tx']
                e    = np.sum(Ka * p * 0.01)
                E.append(e)
            except:
                pass
        energy = np.sum(E)
        energies.append(energy)
        print(f'{solver=} {label=} {energy=}')

    plt.bar(solvers, energies)
    # decorate
    # plt.legend()
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.expected_n_event:0.2f}_{args.metric}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
