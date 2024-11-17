import scipy.special as sp
import numpy as np
import torch

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x.detach().cpu().numpy() / np.sqrt(2))

def qfuncinv(x):
    return torch.sqrt(2) * sp.erfinv(1 - 2 * x.detach().cpu().numpy())

def capacity_awgn(p):
    c = 0.5 * torch.log2(1 + p)
    return c

def normal_approximate_pe(p, # signal interference to noise ratio denominated by N0
                          m, # actual data bit per slot
                          n): # block size of data n_d
    c = capacity_awgn(p)
    R = m / n
    v = 0.5 * p * (p + 2) * (torch.log2(torch.exp(torch.tensor(1.))) / (p + 1)) ** 2
    qinv_pe = (c - R) / torch.sqrt(v / 2 / n)
    pe = qfunc(qinv_pe)
    pe = torch.tensor(pe)
    return pe

def compute_g_db(x_d, y_d, args):
    d = torch.sqrt(x_d ** 2 + y_d ** 2)
    g_db = args.alpha + args.beta * torch.log10(d)
    return g_db

def get_p_tx_watt(gamma_dbm, g_db, p_max=4.8e-8):
    p_tx_dbm = gamma_dbm + g_db
    p_tx_watt = 10 ** (p_tx_dbm / 10) / 1e3 / p_max
    return p_tx_watt

def get_p_tx(gamma, h):
    return gamma * h

def get_sinr_min(n, m, N_0, delta=1e-6, p_md_max=1e-5):
    # grid search with range from 1e-3 - 1000W receiving power
    sinr = torch.tensor([10 ** (-20 + i * 20 / 1000) for i in range(1000)]) / N_0
    sinr_db = 10 * torch.log10(sinr)
    # get pe
    pe = normal_approximate_pe(sinr, m, n)
    # get misdetection probability for all power range
    p_md = (1 - delta) * pe + delta
    idx = torch.min(torch.where(p_md <= p_md_max)[0])
    sinr_min_db = sinr_db[idx]
    sinr_min_db = torch.tensor(sinr_min_db)
    return sinr_min_db

def get_average_group_sic_sinr(pi,  # group target received power, vector of Q elements # in watt
                               xi,  # fraction of devices in group, vector of Q elements
                               Ka,  # number of active devices
                               M,   # number of MIMO antennas
                               N_0, # noise power
                               n_d, # bandwidth for data transmission
                               n_p, # bandwidth for pilot transmission
                               m,   # bitrate
                               ):
    # extract parameter
    Q = len(xi)
    # channel estimation error crude bound, Equation (29)
    sigma_2 = N_0 / (N_0 + n_p * xi * pi)
    # decoding residual power
    I_R = torch.zeros(Q)
    PE  = torch.zeros(Q)
    for q in range(Q):
        pe = normal_approximate_pe(p=pi[q] / N_0, # watt / N0
                                   m=m,
                                   n=n_d)
        PE[q]  = pe
        # TODO: bernoulli random sampling here
        I_R[q] =  Ka * xi[q] * ((1 - pe) * sigma_2[q] * pi[q] + pe * pi[q])
    # interference of lower power signals
    I = torch.zeros(Q)
    for q in range(Q):
        I[q] = Ka * xi[q] * pi[q]
    # signal inference noise ratio
    sinr = torch.zeros(Q)
    for q in range(Q):
        sinr[q] = M * (1 - sigma_2[q]) * pi[q] / \
                  (
                      N_0 + \
                      torch.sum(I_R[:q]) + \
                      torch.sum(I[q:])
                  )
    # convert to db
    sinr_db = 10 * torch.log10(sinr)
    # print(xi, PE)
    return sinr_db

def get_group_sic_sinr(pi,  # group target received power, vector of Q elements # in watt
                       xi,  # fraction of devices in group, vector of Q elements
                       Ka,  # number of active devices
                       M,   # number of MIMO antennas
                       N_0, # noise power
                       n_d, # bandwidth for data transmission
                       n_p, # bandwidth for pilot transmission
                       m,   # bitrate
                       ):
    # extract parameter
    Q = len(xi)
    # channel estimation error crude bound, Equation (29)
    sigma_2 = N_0 / (N_0 + n_p * xi * pi)
    # decoding residual power
    I_R = torch.zeros(Q)
    PE  = torch.zeros(Q)
    for q in range(Q):
        pe = normal_approximate_pe(p=pi[q] / N_0, # watt / N0
                                   m=m,
                                   n=n_d)
        PE[q] = pe
        # sample the error
        n_select_arm_q       = int(Ka * xi[q])
        n_select_arm_q_error = torch.sum(torch.where(torch.rand(n_select_arm_q) < pe)[0])
        I_R[q] =  ((n_select_arm_q - n_select_arm_q_error) * sigma_2[q] * pi[q] + n_select_arm_q_error * pi[q])
    # interference of lower power signals
    I = torch.zeros(Q)
    for q in range(Q):
        n_select_arm_q = int(Ka * xi[q])
        I[q] = n_select_arm_q * pi[q]
    # signal inference noise ratio
    sinr = torch.zeros(Q)
    for q in range(Q):
        sinr[q] = M * (1 - sigma_2[q]) * pi[q] / \
                  (
                      N_0 + \
                      torch.sum(I_R[:q]) + \
                      torch.sum(I[q:])
                  )
    # convert to db
    sinr_db = 10 * torch.log10(sinr)
    # print(xi, PE)
    return sinr_db
