######## Robut calculation of log(1-exp(-a))
#Author: Jonas Kohler
import numpy as np
def log_one_minus_exp_minus_a(a):
	idx = a > np.log(2)
	out = np.empty(a.size, dtype=np.float)
	out[idx] = np.log1p(-np.exp(-a[idx]))
	out[~idx] = np.log(-np.expm1(-a[~idx]))
	return out