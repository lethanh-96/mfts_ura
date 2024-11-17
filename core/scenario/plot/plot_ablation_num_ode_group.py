import matplotlib.pyplot as plt
import os

def plot_ablation_num_ode_group(args):
    # initialize figure size
    fig = plt.figure(figsize=(4, 3))

    # plot from raw dara
    x = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    y = {
        '20' : [0.781, 0.789, 0.797, 0.802, 0.804, 0.807, 0.810, 0.81, 0.811, 0.812],
        '2'  : [0.782, 0.790, 0.7968, 0.8005, 0.802, 0.804, 0.8075, 0.8079, 0.808, 0.809],
        '50' : [0.783, 0.788, 0.795, 0.798, 0.799, 0.802, 0.805, 0.806, 0.8069, 0.8059],
        '100' : [0.785, 0.786, 0.7915, 0.795, 0.796, 0.799, 0.8025, 0.803, 0.805, 0.804],
    }
    styles = ['r-D', 'b-.+', 'g--h', 'k:x']
    for i, solver in enumerate(y.keys()):
        plt.plot(x, y[solver], styles[i], label=solver)

    plt.ylabel('reward')
    plt.xlabel('time slot')
    plt.legend()


    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
