import numpy as np


def series_to_supervised(data, window, forcast_horizon):
    X = []
    y = []
    for i in range(data.shape[0] - window - forcast_horizon + 1):
        X.append(data.iloc[i:i + window])
        y.append(data.iloc[i + window: window + forcast_horizon + i, 0])
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    return X, y


def _calculate_mape(Y_real, Y_pred):
    return np.sum(np.abs(Y_real - Y_pred)) / np.sum(Y_pred)
