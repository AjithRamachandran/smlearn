import numpy as np


def mean_squared_error(y, y_pred) -> float:
    """
    calculates mean squared error.

    :param y: array, true values
    :param y_pred: array, predicted values
    :return: float, mean squared error
    """
    return np.mean(0.5 * np.square(y - y_pred))
