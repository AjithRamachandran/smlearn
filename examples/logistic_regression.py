from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from smlearn.regressors import LogisticRegression
from smlearn.utils import accuracy

X, y = load_digits(n_class=2, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy(y_test, y_pred)
print("Accuracy: ", accuracy)
