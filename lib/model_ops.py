import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Model:
    pipeline = None

    def __init__(self, X, y, random_state: int, test_size=0.2):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.test_predictions = None
        self.test_score = None
        self.train_score = None
        self.coefs = None
        self.test_residuals = None

    def train_score(self):
        return self.model.score(self.X_train, self.y_train)

    def test_score(self):
        return self.model.score(self.X_test, self.y_test)

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.htmlo
    def do_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        print(f"split done, shape X_train is: {self.X_train.shape}")
        print(f"mean of target is: {self.y.mean()}")

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    def apply_linear_regression(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.test_predictions = self.model.predict(self.X_test)
        self.set_metrics()
        self.print_metrics()

    def set_metrics(self):
        self.coefs = pd.DataFrame(self.model.coef_, self.X.columns, columns=['coefficient'])
        self.test_residuals = self.y_test - self.test_predictions

    def print_metrics(self):
        print(f"test set predictions calculated -> (model.test_predictions)")
        print(f"test set mean absolute error is: {mean_absolute_error(self.y_test, self.test_predictions)}")
        print(f"test set mean squared error is: {mean_squared_error(self.y_test, self.test_predictions)}")
        print(f"test set root mean squared error is: {np.sqrt(mean_squared_error(self.y_test, self.test_predictions))}")
        print(f"model coefficients: \n{self.coefs}")
        print(f"test set residuals calculated -> (model.test_residuals)")
        print("-------------------------------------------------")

    def apply_polynomial_regression(self, degree, bias=False):
        polynomial_converter = PolynomialFeatures(degree, bias)
        poly_features = polynomial_converter.fit_transform(self.X)