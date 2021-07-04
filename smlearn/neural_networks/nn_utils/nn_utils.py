import numpy as np


class NNUtils:

    @staticmethod
    def relu(Z):
        ret = np.maximum(0, Z)
        assert(ret.shape == Z.shape)
        return ret

    @staticmethod
    def reverse_relu(d_ip, cache):
        Z = cache

        dZ = np.array(d_ip)
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def sigmoid(Z):
        return 1. / (1. + np.exp(-Z))

    @staticmethod
    def reverse_sigmoid(d_ip, cache):
        Z = cache

        # derivative of sigmoid function is σ(x)(1−σ(x))
        s = 1. / (1. + np.exp(-Z))
        dZ = d_ip * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def softmax(Z):
        return np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0)

    @staticmethod
    def reverse_softmax(d_ip, cache):
        Z = cache
        s = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0)
        dZ = d_ip * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ
