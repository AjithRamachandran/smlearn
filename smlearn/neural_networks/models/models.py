import numpy as np

from smlearn.neural_networks.layers import BaseNNLayer


class BaseNNModel(object):

    def __init__(self) -> None:
        self.n_iters = 0
        self.learning_rate = 0
        self.cost = True
        self.layers = []
        self.layer_dims = []
        self.weights = []
        self.bias = []
        self.caches = []
        self.dweights = []
        self.dbias = []

    def init_params(self):
        np.random.seed(1)
        for l in range(1, len(self.layer_dims)):
            self.weights.append(np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01)
            self.bias.append(np.zeros((self.layer_dims[l], 1)))

    def forward_activation(self, h_ip, weights, bias, activation):
        h_op, cache = activation.activate(h_ip, weights, bias)

        return h_op, cache

    def forward_network(self, X):
        h_op = X.T
        L = len(self.weights)

        for l in range(0, L):
            h_ip = h_op
            h_op, cache = self.forward_activation(h_ip, self.weights[l], self.bias[l], self.layers[l].activation)
            self.caches.append(cache)

        return h_op

    def cost_fn(self, y_pred, y):
        m = y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(np.log(y_pred), y) + np.multiply(np.log(1 - y_pred), (1 - y)))
        cost = np.squeeze(cost)

        return cost

    def backward_activation(self, d_ip, cache, activation):
        dh_ip, dW, db = activation.reverse_activate(d_ip, cache)

        return dh_ip, dW, db

    def backward_network(self, y_pred, y):
        L = len(self.caches)
        y = y.reshape(y_pred.shape)

        dy_pred = - (np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))

        current_cache = self.caches[-1]
        dh_ip_temp, dW_temp, db_temp = self.backward_activation(dy_pred, current_cache, self.layers[-1].activation)
        self.dweights.append(dW_temp)
        self.dbias.append(db_temp)

        for l in reversed(range(L - 1)):
            current_cache = self.caches[l]
            dh_ip_temp, dW_temp, db_temp = self.backward_activation(dh_ip_temp, current_cache,
                                                                    self.layers[l].activation)
            self.dweights.insert(0, dW_temp)
            self.dbias.insert(0, db_temp)

    def update_params(self):
        L = len(self.weights)

        for l in range(0, L):
            self.weights[l] = self.weights[l] - self.learning_rate * self.dweights[l]
            self.bias[l] = self.bias[l] - self.learning_rate * self.dbias[l]

    def add(self, layer):
        if isinstance(layer, BaseNNLayer):
            self.layers.append(layer)
            self.layer_dims.append(layer.n_l)
        else:
            print(layer, "does not belongs to Layer class")

    def compile(self, n_iters=1000, learning_rate=0.01, cost=True) -> None:
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.cost = cost

    def fit(self, X, y) -> None:
        X = X.reshape(X.shape[0], -1)
        self.layer_dims.insert(0, X.shape[1])
        self.init_params()
        for _ in range(self.n_iters):
            nn_op = self.forward_network(X)
            cost = self.cost_fn(nn_op, y)
            self.backward_network(nn_op, y)
            self.update_params()
            self.caches = []
            self.dweights = []
            self.dbias = []

            if self.cost:
                print(cost)

    def predict(self, X):
        h_op = X.T
        for l in range(0, len(self.weights)):
            h_ip = h_op
            h_op, cache = self.forward_activation(h_ip, self.weights[l], self.bias[l], self.layers[l].activation)

        y_pred = np.round(h_op)
        return y_pred


class Sequential(BaseNNModel):

    def __init__(self) -> None:
        super().__init__()
