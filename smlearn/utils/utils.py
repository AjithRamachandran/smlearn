import numpy as np

from itertools import combinations_with_replacement


def mean_squared_error(y, y_pred) -> float:
    """
    calculates mean squared error.

    :param y: array, true values
    :param y_pred: array, predicted values
    :return: float, mean squared error
    """
    return np.mean(0.5 * np.square(y - y_pred))


def polynomial_features(X, degree) -> np.ndarray:
    n_samples, n_features = X.shape

    comb = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
    combinations = [item for sublist in comb for item in sublist]

    poly_x = np.empty((n_samples, len(combinations)))
    for i, index in enumerate(combinations):
        poly_x[:, i] = np.prod(X[:, index], axis=1)

    return poly_x
