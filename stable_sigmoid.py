######## Robut Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########
import numpy as np
def phi(t): #Author: Jonas Kohler
    # logistic function returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out
def log_phi(t):
    # log(Sigmoid): log(1 / (1 + exp(-t)))

    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]=-np.log(1+np.exp(-t[idx]))
    out[~idx]= t[~idx]-np.log(1+np.exp(t[~idx]))
    return out
def log_one_minus_phi(t):
    # log(1-Sigmoid): log(1-1 / (1 + exp(-t)))

    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]= -t[idx]-np.log(1+np.exp(-t[idx]))
    out[~idx]=-np.log(1+np.exp(t[~idx]))
    return out