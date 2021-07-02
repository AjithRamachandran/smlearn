import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from smlearn.regressors import ElasticNet, RidgeRegression, LassoRegression, LinearRegression
from smlearn.utils import mean_squared_error

data = pd.read_csv("datasets/regression.csv")

X = np.atleast_2d(data["X"].values).T
y = data["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

enet_model = ElasticNet(alpha_1=2,
                        alpha_2=2,
                        learning_rate=0.001,
                        n_iterations=2000)
enet_model.fit(X_train, y_train)

rid_model = RidgeRegression(alpha=2,
                            learning_rate=0.001,
                            n_iterations=2000)
rid_model.fit(X_train, y_train)

las_model = LassoRegression(alpha=2,
                            learning_rate=0.001,
                            n_iterations=2000)
las_model.fit(X_train, y_train)

lin_model = LinearRegression(learning_rate=0.001,
                             n_iterations=2000)
lin_model.fit(X_train, y_train)

enet_y_pred = enet_model.predict(X_test)
rid_y_pred = rid_model.predict(X_test)
las_y_pred = las_model.predict(X_test)
lin_y_pred = lin_model.predict(X_test)

mse = mean_squared_error(y_test, enet_y_pred)
print("Mean squared error of Elastic Net Regression: ", mse)
mse = mean_squared_error(y_test, rid_y_pred)
print("Mean squared error of Ridge Regression: ", mse)
mse = mean_squared_error(y_test, las_y_pred)
print("Mean squared error of Lasso Regression: ", mse)
mse = mean_squared_error(y_test, lin_y_pred)
print("Mean squared error of Linear Regression: ", mse)

enet_y_pred_line = enet_model.predict(X)
rid_y_pred_line = rid_model.predict(X)
las_y_pred_line = las_model.predict(X)
lin_y_pred_line = lin_model.predict(X)

cmap = plt.get_cmap('plasma')

m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
plt.plot(366 * X, enet_y_pred_line, color='black', linewidth=1, label="Elastic Net Prediction")
plt.plot(366 * X, rid_y_pred_line, color='red', linewidth=1, label="Ridge Prediction")
plt.plot(366 * X, las_y_pred_line, color='blue', linewidth=1, label="Lasso Prediction")
plt.plot(366 * X, lin_y_pred_line, color='green', linewidth=1, label="Linear Prediction")
plt.suptitle("Elastic Net vs Ridge vs Lasso vs Linear Regression")
plt.xlabel('X')
plt.ylabel('y')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()
