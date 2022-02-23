import numpy as np
from sklearn.metrics import mean_squared_error

def naive_prediction(X):
    y = np.empty(X.shape)
    y[0] = X[0]
    y[1:] = X[:-1]
    return y

def naive_metric(y_true, y_hat, y_naive):
    """
    Good model: has value < 1
    Returns:
    --------
    MSE(y_true, y_hat) / MSE(y_true, y_naive)
    --------
    """
    return mean_squared_error(y_true, y_hat) / mean_squared_error(y_true, y_naive)
