import numpy as np

from smlearn.utils import mean_squared_error, sigmoid


class LogisticRegression:
    """
    logistic regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    :param scale: bool, value that decides if scaling is to be performed or not.
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01, scale: bool = True) -> None:

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.scale = scale
        self.w = self.b = self.loss = []
        self.X_offset = self.y_offset = 0

    def init_weights_and_bias(self, n_features) -> tuple[np.ndarray, int]:
        """
        initiates weights and bias.

        :param n_features: int, number of features.
        :return: array, initialised weights.
        :return: float, initialised bias.
        """
        return np.random.randn(n_features) * 0.01, 0

    @property
    def coef(self) -> np.ndarray:
        """
        returns weight.

        :params: None.
        :return: array, weight.
        """
        return self.w

    @property
    def intercept(self) -> float:
        """
        returns bias.

        :params: None.
        :return: array, bias.
        """
        return self.b

    def preprocess(self, X, y):
        """
        perform scaling on data.

        :param X: array, features.
        :param y: array, true values.
        :return: array, scaled X.
        :return: array, scaled y.
        """
        self.X_offset = np.average(X, axis=0).astype(X.dtype, copy=False)
        self.y_offset = np.average(y, axis=0)

        X, y = X - self.X_offset, y - self.y_offset

        return X, y

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        performs gradient descent algorithm.

        :param X: array, features.
        :param y: array, true values.
        :return: None
        """
        for i in range(self.n_iterations):
            y_pred = sigmoid(np.dot(X, self.w) + self.b)
            self.loss.append(mean_squared_error(y, y_pred))

            partial_w = (1 / X.shape[0]) * (np.dot(X.T, (y_pred - y)))
            partial_b = (1 / X.shape[0]) * (np.sum(y_pred - y))
            self.w -= self.learning_rate * partial_w
            self.b -= self.learning_rate * partial_b

    def fit(self, X, y) -> None:
        """
        calculate the coefficient of the logistic regression model.

        :param X: array, features.
        :param y: array, true values.
        :return: None.
        """
        if self.scale:
            X, y = self.preprocess(X, y)
        self.w, self.b = self.init_weights_and_bias(X.shape[1])
        self.gradient_descent(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions.

        :param X: array, features.
        :return: array, predictions.
        """
        return np.round(sigmoid(np.dot(X, self.w) + self.b)).astype(int)
