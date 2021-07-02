import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from smlearn.regressors import PolynomialRegression
from smlearn.utils import mean_squared_error

data = pd.read_csv("datasets/regression.csv")

X = np.atleast_2d(data["X"].values).T
y = data["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = PolynomialRegression(degree=15, n_iterations=25000)
model.fit(X_train, y_train)

n = len(model.loss)
training, = plt.plot(range(n), model.loss, label="Training Error")
plt.legend(handles=[training])
plt.title("Error Plot")
plt.ylabel("Mean Squared Error")
plt.xlabel("Iterations")
plt.show()

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

y_pred_line = model.predict(X)

cmap = plt.get_cmap("plasma")

m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
plt.plot(366 * X, y_pred_line, color='blue', linewidth=1, label="Prediction")
plt.suptitle("Polynomial Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel("X")
plt.ylabel("y")
plt.legend((m1, m2), ("Training data", "Test data"), loc="lower right")
plt.show()
