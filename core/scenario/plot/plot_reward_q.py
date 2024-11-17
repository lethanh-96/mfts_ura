import matplotlib.pyplot as plt
import os

def plot_reward_q(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))

    # plot from raw dara
    x = [2, 3, 4, 5]
    y = {
        'Centralized': [0.831, 0.834, 0.836, 0.84],
        'TS'         : [0.753, 0.793, 0.795, 0.790],
        'MFTTS'      : [0.75, 0.798, 0.805, 0.804],
        # 'TTS'        : [0.745, 0.792, 0.812, 0.815],
        'Hedge'      : [0.746, 0.784, 0.785, 0.78],
    }
    styles = ['k:*', 'r-D', 'y-s', 'g:x']
    # styles = ['k:*', 'r-D', 'y-s', 'm-h', 'g:x']
    for i, solver in enumerate(y.keys()):
        plt.plot(x, y[solver], styles[i], label=solver)

    plt.ylabel('reward')
    plt.xlabel('$Q$ - number of power level')
    plt.legend(ncol=2)

    plt.xticks(x, x)
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
