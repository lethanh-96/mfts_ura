from torch.autograd import Variable
import torch

def compute_gamma_opt():
    def f(x):
        return torch.pow(1 - torch.exp(-x), 100)
    def df(x):
        return 100 * torch.exp(-100*x) * torch.pow(torch.exp(x) - 1, 99)
    x = torch.linspace(-10, 10, 1000)
    fd = torch.abs(x * f(x) - df(x))
    mask = torch.logical_and(torch.logical_not(torch.isnan(fd)), fd < 1e-6)
    idx = torch.where(mask)[0][0]
    gamma_opt_db = x[idx]
    gamma_opt = 10 ** (gamma_opt_db / 10)
    return gamma_opt

def compute_opt_benamor_p_tx(h_db, d, gamma_opt, args):
    # extract parameter
    N_0   = args.noise_power
    K     = len(h_db)
    alpha = args.max_frequency_reuse
    h     = 10 ** (h_db / 10)
    M     = args.n_antenna
#     # compute average power, binary search for best smallest base power
#     # perform in base_station
#     rx_power_max = 10
#     rx_power_min = -100
#     avg_p_rx_N0_db = torch.tensor(rx_power_min) # torch.tensor((args.rx_power_max + args.rx_power_min) / 2)
#     tmp = torch.tensor(0)
#     eps = 1000
#     while eps > 1e-1:
#         avg_p_rx  = 10 ** (tmp / 10) * N_0
#         # compute own interference for each device
#         total_inference = gamma_opt * ((alpha * avg_p_rx * (1 - d)))
#         # compute own power coefficient for each device
#         p_tx = total_inference * h
#         violated_idx = compute_sinr_constraint(p_tx, h_db, gamma_opt, args)
#         if len(violated_idx) > 0: # smallest user failed, try increasing
#             rx_power_min = tmp
#             tmp = (tmp + rx_power_max) / 2
#         else: # smallest user decoded, try reducing
#             rx_power_max = tmp
#             tmp = (tmp + rx_power_min) / 2
#         eps = rx_power_max - rx_power_min
    tmp = torch.tensor(args.rx_power_avg)
    eps = 0
    # finally compute the minimum transmit power
    # compute own interference for each device
    avg_p_rx  = 10 ** ((tmp + eps) / 10) * N_0
    total_inference = gamma_opt * ((alpha * avg_p_rx * (1 - d)))
    # compute own power coefficient for each device
    p_tx = total_inference * h
    # p_rx = 10 * torch.log10(p_tx / h / N_0)
    # violated_idx = compute_sinr_constraint(p_tx, h_db, gamma_opt, args)
    # sinr = M * p_rx / (sum_interference(p_rx) + N_0)
    return p_tx

def sum_lower_interference(x):
    # sort
    x, idx = torch.sort(x)
    # decode
    _, idx_decode = torch.sort(idx)
    # cumulative sum
    x = torch.cumsum(x, dim=0)
    x = torch.cat((torch.tensor([0.]), x[:-1]))
    # interference
    I = x[idx_decode]
    return I

def sum_higher_interference(x, err):
    # sort
    x, idx = torch.sort(x)[::-1]
    # decode
    _, idx_decode = torch.sort(idx)
    # cumulative sum
    x = x * err
    x = torch.cumsum(x, dim=0)
    x = torch.cat((torch.tensor([0.]), x[:-1]))
    # interference
    I = x[idx_decode]
    return I

def compute_sinr_constraint(p_tx, h_db, gamma_opt, args):
    # extract parameter
    N_0   = args.noise_power
    K     = len(h_db)
    alpha = args.max_frequency_reuse
    M     = args.n_antenna
    h     = 10 ** (h_db / 10)
    n_p   = args.n_pilot_bit
    # channel estimation error crude bound, Equation (29)
    sigma_2 = N_0 / (N_0 + n_p * p_tx)
    #
    p_rx = p_tx / h
    sinr = M * p_rx / (sum_higher_interference(p_rx, sigma_2) + sum_lower_interference(p_rx) + N_0)
    violated_idx = torch.where(sinr < gamma_opt)[0]
    return violated_idx
