import numpy as np

#progressive averaging
def prog_avg(x):
    s = np.cumsum(x, dtype=float)
    return np.array([s[i]/(i+1) for i in range(len(x))])

#sliding averaging
def win_avg(x, w, compensate=False):
    cum_sum = np.cumsum(x, dtype=float)
    valid_y = np.true_divide(np.subtract(cum_sum[w:], cum_sum[:-w]), float(w))
    if compensate:
        begin_y = np.true_divide(cum_sum[:w], range(1,w+1))
    else:
        begin_y = np.true_divide(cum_sum[:w], float(w))
    return np.append(begin_y, valid_y)
	
