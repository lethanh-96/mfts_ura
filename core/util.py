import argparse
import torch
import os

base_folder = os.path.dirname(os.path.dirname(__file__))

def create_directories(args):
    for d in [args.figure_dir, args.csv_dir, args.model_dir,
              args.approximator_trace_dir, args.simulator_trace_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

def set_tuned_parameter(args):
    if args.solver == 'mabmfg':
        args.rx_power_avg = 1
        args.gamma_opt    = 0.9
    if args.solver in ['ts', 'tts', 'random', 'hedge', 'greedy', 'weighted_random', 'centralized']:
        args.rx_power_avg = -5
        args.rx_power_gap = 1

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    # scenario
    parser.add_argument('--scenario', type=str, default='simulate')
    parser.add_argument('--mode', type=str, default='test')
    # solver
    parser.add_argument('--solver', type=str, default='ts')
    # hedge solver
    parser.add_argument('--exploration_rate', type=float, default=0.2)
    parser.add_argument('--smoothing_coeff', type=int, default=30)
    # ts solver
    parser.add_argument('--exp_coeff', type=float, default=0.75)
    # centralized solver
    parser.add_argument('--centralized_step', type=int, default=1)
    parser.add_argument('--max_simulation_time', type=int, default=1200)
    # mabmfg solver
    parser.add_argument('--max_frequency_reuse', type=int, default=5)
    parser.add_argument('--epsilon_0', type=float, default=0.2)
    parser.add_argument('--t_mab', type=float, default=100)
    parser.add_argument('--gamma_opt', type=float, default=0.9)
    # approximator solver
    parser.add_argument('--n_ode_group', type=int, default=20)
    parser.add_argument('--n_approximator_sample', type=int, default=500)
    # mfdqn solver
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=float, default=1000)
    parser.add_argument('--tau', type=int, default=0.005)
    parser.add_argument('--lr', type=int, default=1e-4)
    # traffic model
    parser.add_argument('--traffic_model', type=str, default='sppp')
    # SPPP traffic model
    parser.add_argument('--expected_n_device', type=float, default=10000)
    parser.add_argument('--expected_n_event', type=float, default=6)
    parser.add_argument('--r_min', type=float, default=0.25)
    parser.add_argument('--r_max', type=float, default=1.0)
    parser.add_argument('--p_ru', type=float, default=1/6000) # probability of regular update
    parser.add_argument('--q', type=float, default=0.995) # probability of stay in alarm state
    parser.add_argument('--r_d', type=float, default=5e-3) # average radius of being affected by event UMA - large scale fading
    parser.add_argument('--alpha', type=float, default=128.1)
    parser.add_argument('--beta', type=float, default=36.7)
    parser.add_argument('--sigma2shadow', type=float, default=4.0)
    # UMA - power allocation
    parser.add_argument('--n_arm', type=int, default=5)
    parser.add_argument('--n_antenna', type=int, default=100)
    parser.add_argument('--noise_power', type=float, default=10 ** (-20.4) * 1.4e6)
    parser.add_argument('--rx_power_avg', type=float, default=-5)
    parser.add_argument('--rx_power_gap', type=float, default=1)
    parser.add_argument('--n_data_bit', type=int, default=2048)
    parser.add_argument('--n_pilot_bit', type=int, default=1152)
    parser.add_argument('--bit_rate', type=int, default=100)
    parser.add_argument('--p_tx_max', type=int, default=0.2) # watts
    # experimental setup
    parser.add_argument('--n_step', type=int, default=1 * 60 * 100)
    parser.add_argument('--seed', type=int, default=0)
    # plot
    parser.add_argument('--metric', type=str, default='avg_reward')
    parser.add_argument('--n_plot_step', type=int, default=1000)
    # I/O
    parser.add_argument('--figure_dir', type=str, default=f'{base_folder}/data/figure')
    parser.add_argument('--csv_dir', type=str, default=f'{base_folder}/data/csv')
    parser.add_argument('--model_dir', type=str, default=f'{base_folder}/data/model')
    parser.add_argument('--approximator_trace_dir', type=str, default=f'{base_folder}/data/approximator_trace')
    parser.add_argument('--simulator_trace_dir', type=str, default=f'{base_folder}/data/simulator_trace')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_trace', action='store_true')
    parser.add_argument('--tuned_parameter', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--skip', action='store_true')
    # hardware
    parser.add_argument('--device', type=str, default='cpu')
    # parse args
    args = parser.parse_args()
    # set tuned parameters
    if args.tuned_parameter:
        set_tuned_parameter(args)
    # create directory if not exists
    create_directories(args)
    # set default hardware
    torch.set_default_device(args.device)
    # benamor solver
    args.n_group = int((args.n_pilot_bit + args.n_data_bit) / args.bit_rate * 0.64)
    # rx power min/max derivation
    args.rx_power_min = args.rx_power_avg - (args.n_arm - args.rx_power_gap)
    args.rx_power_max = args.rx_power_avg + (args.n_arm - args.rx_power_gap)
    # for mfdql
    args.n_observation = 1 + args.n_arm
    args.n_action      = args.n_arm
    return args

def print_args(args):
    pass
