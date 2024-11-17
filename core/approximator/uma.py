from sympy import erf, erfinv, log, sqrt
import numpy as np

def qfunc(x):
    return 0.5 - 0.5 * erf(x / np.sqrt(2))

def capacity_awgn(p):
    c = 0.5 * log(1 + p, 2)
    return c

def normal_approximate_pe(p, # signal interference to noise ratio denominated by N0
                          m, # actual data bit per slot
                          n): # block size of data n_d
    c = capacity_awgn(p)
    R = m / n
    v = 0.5 * p * (p + 2) * (np.log2(np.exp(1)) / (p + 1)) ** 2
    qinv_pe = (c - R) / sqrt(v / 2 / n)
    pe = qfunc(qinv_pe)
    return pe

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
    Q = len(pi)
    # channel estimation error crude bound, Equation (29)
    sigma_2 = N_0 / (N_0 + n_p * xi * pi)
    # decoding residual power
    I_R = []
    for q in range(Q):
        pe = normal_approximate_pe(p=pi[q] / N_0, # watt / N0
                                   m=m,
                                   n=n_d)
        I_R.append(Ka * xi[q] * ((1 - pe) * sigma_2[q] * pi[q] + pe * pi[q]))
    I_R = np.array(I_R)
    # interference of lower power signals
    I = []
    for q in range(Q):
        I.append(Ka * xi[q] * pi[q])
    I = np.array(I)
    # signal inference noise ratio
    sinr = []
    for q in range(Q):
        sinr_q = M * (1 - sigma_2[q]) * pi[q] / \
                  (
                      N_0 + \
                      np.sum(I_R[:q]) + \
                      np.sum(I[q:])
                  )
        sinr.append(sinr_q)
    # convert to db
    sinr_db = [10 * log(sinr[q], 10) for q in range(Q)]
    return sinr_db
