import matplotlib.pyplot as plt
import numpy as np
import os

def plot_legend(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))

    x = np.random.rand(10)
    y = np.random.rand(10)
    fig, ax = plt.subplots()

    styles  = ['g:x',  'r-D', 'y-s', 'k:*']
    solvers = ['hedge', 'ts', 'tts', 'greedy']
    labels  = ['Hedge', 'TS', 'MFTTS', 'Centralized']

    for style, label in zip(styles, labels):
        ax.plot(x, y, style, label=label)

    ax.legend(ncol=6)

    legend_fig = plt.figure(figsize=(8, 6))
    handles, labels = ax.get_legend_handles_labels()
    legend_fig.legend(handles, labels, ncol=6)

    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    legend_fig.tight_layout()
    legend_fig.savefig(path)
    plt.clf()
