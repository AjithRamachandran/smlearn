import numpy as np

from smlearn.utils import mean_squared_error


class Regression(object):
    """
    Base regression model. Models the relationship between a scalar dependent variable y and the independent
    variables X.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations: int, learning_rate: float) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = []

    """
    initiates weights and bias.

    :param n_features: int, number of features
    :return: array, initialised weights
    :return: float, initialised bias
    """

    def init_weights_and_bias(self, n_features) -> tuple[np.ndarray, int]:
        return np.random.randn(n_features), 0

    """
    returns weight.

    :params: None
    :return: array, weight
    """

    @property
    def coef(self) -> np.ndarray:
        return self.w

    """
    returns bias.

    :params: None
    :return: array, bias
    """

    @property
    def intercept(self) -> float:
        return self.b

    """
    performs gradient descent algorithm.

    :param X: array, features
    :param y: array, true values
    :return: None
    """

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.w) + self.b
            self.loss.append(mean_squared_error(y, y_pred))

            partial_w = (1 / X.shape[0]) * ((np.dot(X.T, (y_pred - y))) + self.regularization(self.w))
            partial_b = (1 / X.shape[0]) * (np.sum(y_pred - y))
            self.w -= self.learning_rate * partial_w
            self.b -= self.learning_rate * partial_b

    """
    calculate the coefficient of the linear regression model.

    :param X: array, features
    :param y: array, true values
    :return: None
    """

    def fit(self, X, y) -> None:
        self.w, self.b = self.init_weights_and_bias(X.shape[1])
        self.gradient_descent(X, y)

    """
    Makes predictions using the line equation.

    :param X: array, features
    :return: array, predictions
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b
