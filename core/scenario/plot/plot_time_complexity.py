import matplotlib.pyplot as plt
import os

def plot_time_complexity(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))

    # plot from raw dara
    x = [2, 3, 4, 5]
    y = {
        'Hedge'         : [0.050, 0.051, 0.052, 0.053],
        'TS or MFTTS': [0.060, 0.061, 0.062, 0.063],
        'Centralized'   : [60, 270, 1050, 6000],
    }
    styles = ['g:x', 'r-D', 'k:*']
    for i, solver in enumerate(y.keys()):
        plt.plot(x, y[solver], styles[i], label=solver)

    plt.ylabel('decision time (ms)')
    plt.xlabel('$Q$ - number of power level')
    plt.legend()
    plt.yscale('log')
    plt.xticks(x, x)
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
