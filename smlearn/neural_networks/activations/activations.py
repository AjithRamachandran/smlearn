import numpy as np

from smlearn.neural_networks.nn_utils import NNUtils


class BaseActivation(object):

    def __init__(self) -> None:
        pass

    def activate(self, h_ip, weights, bias):
        input_cache = (h_ip, weights, bias)
        Z = weights.dot(h_ip) + bias
        h_op = self.forward_activation(Z)
        activation_cache = Z
        caches = (input_cache, activation_cache)

        return h_op, caches

    def reverse_activate(self, d_ip, cache):
        input_cache, activation_cache = cache
        dZ = self.backward_activation(d_ip, activation_cache)
        h_ip_prev, W, b = input_cache
        m = h_ip_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, h_ip_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dh_ip_prev = np.dot(W.T, dZ)

        return dh_ip_prev, dW, db


class Relu(BaseActivation):

    def __init__(self) -> None:
        self.forward_activation = NNUtils.relu
        self.backward_activation = NNUtils.reverse_relu
        super().__init__()


class Sigmoid(BaseActivation):

    def __init__(self) -> None:
        self.forward_activation = NNUtils.sigmoid
        self.backward_activation = NNUtils.reverse_sigmoid
        super().__init__()


class Softmax(BaseActivation):

    def __init__(self) -> None:
        self.forward_activation = NNUtils.softmax
        self.backward_activation = NNUtils.reverse_softmax
        super().__init__()
