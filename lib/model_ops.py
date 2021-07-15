import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class DfOps:

    def __init__(self, X, y, degree: int, random_state: int, test_size=0.2):

    # # Linear regression model
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45, test_size=0.2)
    # model = make_pipeline(PolynomialFeatures(degree=4), linear_model.LinearRegression())
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)