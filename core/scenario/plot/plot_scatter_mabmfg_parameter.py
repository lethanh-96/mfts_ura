import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import traceback
import os

def plot_scatter_mabmfg_parameter(args):
    # list of parameters to compare
    rx_power_avgs = np.arange(-15, 5 + 1, 1)
    args.csv_dir  = '../data/csv_mabmfg_parameter'
    expected_n_events = [6, 7, 8, 9, 10]
    Y1, Y2 = [], []
    # extract two metric
    for rx_power_avg in rx_power_avgs:
        y1_, y2_ = [], []
        for expected_n_event in expected_n_events:
            try:
                path = os.path.join(args.csv_dir, f'{rx_power_avg:0.1f}_{expected_n_event:0.1f}.csv')
                df   = pd.read_csv(path)
                y1   = np.mean(df['avg_p_tx'].to_numpy())
                y2   = np.mean(df['n_violated'].to_numpy())
                y1_.append(y1)
                y2_.append(y2)
            except:
                print(f'[!] error: {rx_power_avg}_{expected_n_event:0.1f}.csv')
        y1, y2 = np.mean(y1_), np.mean(y2_)
        Y1.append(y1)
        Y2.append(y2)
    # scatter
    plt.scatter(Y1, Y2)
    plt.xlabel('avg_p_tx')
    plt.ylabel('n_violated')
    # save the plot
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
