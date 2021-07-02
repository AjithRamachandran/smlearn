from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from smlearn.regressors import LinearRegression
from smlearn.utils import mean_squared_error

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(method="ols")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)
