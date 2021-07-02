import numpy as np

from smlearn.utils import mean_squared_error, polynomial_features


class L1Regularization:
    """
    L1 Regularization class.

    :param l1_alpha: float, alpha for L1 Regularization.
    """

    def __init__(self, l1_alpha) -> None:
        self.l1_alpha = l1_alpha

    def __call__(self, w):
        """
        performs L1 Regularization for given weights.

        :param w: array, weights to apply Regularization on.
        """
        return self.l1_alpha * np.abs(w).sum()


class L2Regularization:
    """
    L2 Regularization class.

    :param l2_alpha: float, alpha for L2 Regularization.
    """

    def __init__(self, l2_alpha) -> None:
        self.l2_alpha = l2_alpha

    def __call__(self, w) -> float:
        """
        performs L2 Regularization for given weights.

        :param w: array, weights to apply Regularization on.
        """
        return self.l2_alpha * np.square(w).sum()


class L1L2Regularization:
    """
    Combined L1 and L2 Regularization class.

    :param l1_alpha: float, alpha for L1 Regularization.
    :param l2_alpha: float, alpha for L2 Regularization.
    """

    def __init__(self, l1_alpha, l2_alpha) -> None:
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha

    def __call__(self, w) -> float:
        """
        performs L1L2 Regularization for given weights.

        :param w: array, weights to apply Regularization on.
        """
        l1l2_ratio = self.l2_alpha / (self.l1_alpha + self.l2_alpha)
        return l1l2_ratio * np.square(w).sum() + (1 - l1l2_ratio) * np.abs(w).sum()


class Regression(object):
    """
    Base regression model. Models the relationship between a scalar dependent variable y and the independent
    variables X.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations: int, learning_rate: float, method: str, scale: bool) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.scale = scale
        self.method = method
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
            y_pred = np.dot(X, self.w) + self.b
            self.loss.append(mean_squared_error(y, y_pred))

            partial_w = (1 / X.shape[0]) * ((np.dot(X.T, (y_pred - y))) + self.regularization(self.w))
            partial_b = (1 / X.shape[0]) * (np.sum(y_pred - y))
            self.w -= self.learning_rate * partial_w
            self.b -= self.learning_rate * partial_b

    def ordinary_least_squares(self, X: np.ndarray, y: np.ndarray):
        """
        performs ordinary least squares algorithm.

        :param X: array, features.
        :param y: array, true values.
        :return: None.
        """

        self.w = np.linalg.lstsq(X, y, rcond=-1)[0]
        self.b = self.y_offset - np.dot(self.X_offset, self.w)

    def fit(self, X, y) -> None:
        """
        calculate the coefficient of the linear regression model.

        :param X: array, features.
        :param y: array, true values.
        :return: None.
        """
        if self.scale:
            X, y = self.preprocess(X, y)
        self.w, self.b = self.init_weights_and_bias(X.shape[1])
        if self.method == "gd":
            self.gradient_descent(X, y)
        elif self.method == "ols":
            self.ordinary_least_squares(X, y)
        else:
            raise NotImplemented

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the line equation.

        :param X: array, features.
        :return: array, predictions.
        """
        return np.dot(X, self.w) + self.b


class LinearRegression(Regression):
    """
    linear regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01,
                 method: str = "gd", scale: bool = True) -> None:
        self.regularization = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate,
                                               method=method,
                                               scale=scale)


class LassoRegression(Regression):
    """
    lasso regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    :param alpha: float, value of lambda (λ).
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01, alpha: float = 1.0,
                 method: str = "gd", scale: bool = True) -> None:
        self.regularization = L1Regularization(l1_alpha=alpha)
        super(LassoRegression, self).__init__(n_iterations=n_iterations,
                                              learning_rate=learning_rate,
                                              method=method,
                                              scale=scale)


class RidgeRegression(Regression):
    """
    ridge regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    :param alpha: float, value of lambda (λ).
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01, alpha: float = 1.0,
                 method: str = "gd", scale: bool = True) -> None:
        self.regularization = L2Regularization(l2_alpha=alpha)
        super(RidgeRegression, self).__init__(n_iterations=n_iterations,
                                              learning_rate=learning_rate,
                                              method=method,
                                              scale=scale)


class PolynomialRegression(Regression):
    """
    polynomial regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01, degree: int = 2,
                 method: str = "gd", scale: bool = True) -> None:
        self.degree = degree
        self.regularization = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                   learning_rate=learning_rate,
                                                   method=method,
                                                   scale=scale)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = polynomial_features(X, self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = polynomial_features(X, self.degree)
        return super(PolynomialRegression, self).predict(X)


class ElasticNet(Regression):
    """
    elastic net regression model.

    :param n_iterations: float, number of training iterations the algorithm will tune the weights for.
    :param learning_rate: float, step length that will be used when updating the weights.
    :param alpha_1: float, value of lambda_1 (λ1).
    :param alpha_2: float, value of lambda_2 (λ2).
    """

    def __init__(self, n_iterations: int = 300, learning_rate: float = 0.01,
                 alpha_1: float = 1.0, alpha_2: float = 1.0, method: str = "gd", scale: bool = True) -> None:
        self.regularization = L1L2Regularization(l1_alpha=alpha_1, l2_alpha=alpha_2)
        super(ElasticNet, self).__init__(n_iterations=n_iterations,
                                         learning_rate=learning_rate,
                                         method=method,
                                         scale=scale)
