import numpy as np
import h5py

from smlearn.neural_networks.models import Sequential
from smlearn.neural_networks.activations import Relu, Sigmoid
from smlearn.neural_networks.layers import Dense
from smlearn.utils import accuracy


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def get_model():
    model = Sequential()
    model.add(Dense(20, Relu()))
    model.add(Dense(10, Relu()))
    model.add(Dense(5, Relu()))
    model.add(Dense(1, Sigmoid()))

    model.compile(n_iters=1000, cost=False)

    return model


def main():
    train_x, train_y, test_x, test_y, classes = load_data()

    # Reshape the training and test examples
    train_x = train_x.reshape(train_x.shape[0], -1)  # The "-1" makes reshape flatten the remaining dimensions
    test_x = test_x.reshape(test_x.shape[0], -1)

    print(train_x.shape)
    print(train_y.shape)

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x / 255.
    test_x = test_x / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("train_y's shape: " + str(train_y.shape))
    print("test_y's shape: " + str(test_y.shape))

    model = get_model()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    assert (pred_y.shape == test_y.shape)
    print(accuracy(test_y[0], pred_y[0]))


if __name__ == "__main__":
    main()
