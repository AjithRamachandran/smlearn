from smlearn.neural_networks.activations import BaseActivation

class BaseNNLayer(object):

    def __init__(self, n_l, activation) -> None:
        self.n_l = n_l
        if(isinstance(activation, BaseActivation)):
            self.activation = activation
        else:
            print("Not a Valid activation")

class Dense(BaseNNLayer):

    def __init__(self, n_l, activation) -> None:
        super().__init__(n_l, activation)
