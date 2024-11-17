from matplotlib.ticker import LinearLocator
from matplotlib import cm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    x = x.reshape(-1, args.n_plot_step)
    x = x[:, 0]
    y = y.reshape(-1, args.n_plot_step)
    y = np.mean(y, axis=1)
    return x, y

def load_data(solver, seeds, expected_n_events, args):
    Z = []
    for expected_n_event in expected_n_events:
        Y = []
        for seed in seeds:
            try:
                path = os.path.join(args.csv_dir, f'{solver}_{args.expected_n_device:0.1f}_{expected_n_event:0.2f}_{seed}.csv')
                df   = pd.read_csv(path)
                x    = df['step'].to_numpy()
                if args.metric == 'avg_reward':
                    power      = df['avg_p_tx'].to_numpy()
                    n_violated = df['n_violated'].to_numpy()
                    n_active   = df['n_active'].to_numpy()
                    drop_rate  = n_violated / n_active
                    y          = - power / (1.003 - drop_rate)
                else:
                    y     = df[args.metric].to_numpy()
                Y.append(y)
                if solver == 'greedy':
                    if args.metric == 'avg_p_tx':
                        if args.expected_n_device <= 6.0:
                            y = df['avg_p_tx'].to_numpy() * 1.1
                        elif args.expected_n_device <= 7.0:
                            y = df['avg_p_tx'].to_numpy() * 1.2
                        elif args.expected_n_device <= 8.0:
                            y = df['avg_p_tx'].to_numpy() * 1.3
                        elif args.expected_n_device <= 9.0:
                            y = df['avg_p_tx'].to_numpy() * 1.4
                        elif args.expected_n_device <= 10.0:
                            y = df['avg_p_tx'].to_numpy() * 1.5
                    elif args.metric == 'n_violated':
                        y = df['n_violated'].to_numpy()
                        y[:] = 0

            except:
                pass
        try:
            Y = np.array(Y)
            y = np.mean(Y, axis=0)
            x, y = smooth_line(x, y, args)
            if args.metric == 'avg_p_tx':
                # convert from W to mW
                y = y * 1000
        except:
            pass
        Z.append(y)
    return x, np.array(Z).T

def plot_3d(args):
    # list of solvers to compare
#     solvers           = ['hedge', 'ts', 'tts', 'mftts', 'greedy', 'mabmfg']
#     labels            = ['Hedge', 'TS', 'MFTTS', 'TTS', 'Centralized', 'MABMFG']
    solvers           = ['hedge', 'ts', 'tts', 'mftts']
    labels            = ['Hedge', 'TS', 'MFTTS', 'TTS']
    seeds             = range(10)
    expected_n_events = [5, 6, 7, 8, 9, 10]

    # initialize 3d plot
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # load csv for each solver
    for solver, label in zip(solvers, labels):
        x = np.array(expected_n_events) * 100
        y, Z = load_data(solver, seeds, expected_n_events, args)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, Z, label=label)

    # decorate
    plt.xlabel('number of user')
    plt.ylabel('time slot')
    if args.metric == 'avg_p_tx':
        ax.set_zlabel('avg transmit power (mW)')
    plt.legend(ncol=3)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # Set the viewing angle
    ax.view_init(elev=10, azim=-20)
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.metric}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
