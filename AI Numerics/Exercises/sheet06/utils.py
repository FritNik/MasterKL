import numpy as np



def slog(X, tol=1e-300):
    """Safe log function (avoids underflow for small values)

    ### Parameters
    1. X : str
        - input you wish to take the log of

    ### Returns
    - log(X)
        - the log of the maximum between X and tol
    """
    return np.log(np.clip(X, a_min=tol, a_max=None))

def make_worm(D: int, T: int, sigma: float, rng=np.random.default_rng()):
    tn = np.ceil(T / 3).astype(int)
    X = rng.random([D, tn * 3])

    y = np.zeros(tn * 3)

    for k in range(0, 3):
        X[0:2, tn * k : tn * (k + 1)] = rng.multivariate_normal(
            [(k - 1) / (4), 0], np.diag([sigma * 0.1, sigma]), tn
        ).T
        y[tn * (k) : tn * (k + 1)] = (k + 1) % 2

    alpha = (45 * np.pi) / 180
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    X[0:2, :] = rot @ X[0:2, :]

    # Min Max scaling of the feature dimensions
    X = X - np.reshape(np.min(X, 1), (D, 1))
    X = X / np.reshape(np.max(X, 1), (D, 1))
    return X.T, y
