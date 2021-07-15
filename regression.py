from lib import df_ops as dop
import numpy as np
import sklearn.datasets as datasets
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from lib.model_ops import Model as mdl


def run():
    df_houses = pd.read_csv('assets/temp_output.csv', sep=',', index_col=0)
    dfo = dop.DfOps(df_houses, 50)
    # drop
    # dropping rows for subtype
    # get list of strings that have values below treshold
    column = "property_subtype"
    unwanted_property = dfo.strings_in_column_below_treshold_are(column, 0.3)
    dfo.drop_rows_having_strings_in_column(column, unwanted_property)

    print(dfo.df.shape)
    # Dropping properties that have more then 12 bedrooms
    dfo.drop_rows_smaller_than_treshold("bedrooms", 12.0)

    # Dropping properties that cost more then 2000000
    dfo.drop_rows_smaller_than_treshold("price", 2000000)

    # Dropping properties with more then 500m2
    dfo.drop_rows_smaller_than_treshold("area", 500.0)

    # reindex after dropping rows
    dfo.reindex()
    print(dfo.df.shape)
    # Area & Bedrooms features vs Price model
    # Features
    features = ["area", "bedrooms"]
    target = "price"
    X = dfo.df[features]
    y = dfo.df[target]

    # Linear regression model
    # creating a mdl instance creates the train and test variables at initialization
    linear_model = mdl(X, y, 45)  # default test_size=0.2
    linear_model.apply_linear_regression()

    # poly_model = mdl(X, y, 45)
    # poly_model.apply_polynomial_regression(degree=4)
    # todo: numeric state of house

    """
    # Model score
    train_score = model.score(X_train, y_train)
    train_score = train_score * 100
    test_score = model.score(X_test, y_test)
    test_score = test_score * 100
    print(f'Train score (dof = 4) features area and bedrooms: {train_score} %')
    print(f'Test score (dof = 4) features area and bedrooms: {test_score} %\n')
    
    # Train score (dof = 4) features area and bedrooms: 47.80220051078344 %
    # Test score (dof = 4) features area and bedrooms: 52.6614624863633 %
    """
    # graphics
    corr = sns.heatmap(dfo.df.corr(), linewidths=0.4, cmap="YlGnBu")
    corr.set_title('Correlation between all the house features')

if __name__ == '__main__':
    run()
