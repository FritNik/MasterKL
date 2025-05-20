import numpy as np


def make_data(D, T, duplicate=False, quadratic=False, noise=0.1):
    X = np.ones([T, D + 1])
    X[:, 1:] = np.random.rand(T, D)

    if duplicate:
        X = np.column_stack((X, X[:, -1]))

    theta_true = 5 * (np.random.rand(X.shape[1], 1) - 0.5)
    if quadratic:
        y = (X**2 + noise * np.random.randn(*X.shape)) @ theta_true
    else:
        y = (X + noise * np.random.randn(*X.shape)) @ theta_true
    return X, y, theta_true


def evaluate_regression_plot(theta, low=0, high=1, res=20):
    x_plot = np.ones([res, theta.shape[0]])
    x_plot[:, 1:] = np.linspace(low, high, res)[:, np.newaxis]
    return x_plot[:, 1], x_plot @ theta

def evaluate_regression_plot_2D(theta, low=0, high=1, res=20):
    x_plot = np.ones((res*res, theta.shape[0]))

    single = np.linspace(low,high,res)
    xx1,xx2 = np.meshgrid(single, single)
    x_plot[:, 1:] = np.column_stack((xx1.flatten(), xx2.flatten()))

    return x_plot, x_plot @ theta, single, (x_plot @ theta).reshape(res,res)
