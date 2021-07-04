import numpy as np

from sklearn import model_selection
from sklearn.datasets import load_digits

from smlearn.neural_networks.models import Sequential
from smlearn.neural_networks.activations import Relu, Sigmoid
from smlearn.neural_networks.layers import Dense
from smlearn.utils import accuracy

X, y = load_digits(n_class=2, return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=23)

y_train, y_test = np.array(y_train, ndmin=2), np.array(y_test, ndmin=2)

model = Sequential()
model.add(Dense(5, Relu()))
model.add(Dense(1, Sigmoid()))

model.compile(n_iters=100, cost=False)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
assert(y_pred.shape == y_test.shape)

print("Accuracy: ", accuracy(y_test[0], y_pred[0]))
