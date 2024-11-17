import matplotlib.pyplot as plt
import os

def plot_ablation_transfer(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))

    # plot from raw dara
    x = [2, 3, 4, 5]
    y = {
        'TS': [0.753, 0.796, 0.795, 0.790],
        'TS $\\mu$': [0.752, 0.795, 0.794, 0.790],
        'TS $(\\mu, c)$': [0.746, 0.794, 0.811, 0.812],
    }
    styles = ['r-D', 'c:+', 'm--h']
    for i, solver in enumerate(y.keys()):
        plt.plot(x, y[solver], styles[i], label=solver)

    plt.ylabel('reward')
    plt.xlabel('$Q$ - number of power level')
    plt.legend()

    plt.xticks(x, x)
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
