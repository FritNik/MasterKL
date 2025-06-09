import numpy as np
from scipy.sparse import identity, diags_array

def gen_example(T=1000, n_switches=6, noise=0.01):

    X = np.zeros(T)

    l = T//(n_switches+1)

    for n in range(n_switches+1):
        if np.mod(n,2)!=0:
            X[l*n:l*(n+1)] = 1

    X_noisy = X + (np.random.rand(T)-0.5) * noise
    return X, X_noisy


def get_laplacian_grid_1d(T):
    a = np.ones(T)*2
    a[0]=1
    a[-1] = 1
    return diags_array(
        [a, -np.ones(T),-np.ones(T)],
        offsets=[0,1,-1]
    )
