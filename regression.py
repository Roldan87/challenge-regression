from lib import dfops as dop
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


def run():
    df_houses = pd.read_csv('assets/temp_output.csv', sep=',', index_col=0)
    dfo = dop.DfOps(df_houses, 50)
    # get list of strings that have values below treshold
    column = "property_subtype"
    unwanted_property = dfo.strings_in_column_below_treshold_are(column, 0.3)

    # confirm row count for rows we're going to drop
    # print(dfo.count_rows_having_strings_in_column(column, unwanted_property))  # 70
    
    # now drop them
    dfo.drop_rows_having_strings_in_column(column, unwanted_property)
    print(dfo.df.shape[0])  # row count 9988, rows have been dropped
    # print(dfo.one_hot_encode(""))

if __name__ == '__main__':
    run()






# Area & Bedrooms features vs Price model

# Features
x = df[['area', 'bedrooms']]
y = df['price']

# Linear regression model
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)
model = make_pipeline(PolynomialFeatures(degree=4), linear_model.LinearRegression())
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# Model score
train_score = model.score(X_train, y_train)
train_score = train_score * 100
test_score = model.score(X_test, y_test)
test_score = test_score * 100
print(f'Train score (dof = 4) features area and bedrooms: {train_score} %')
print(f'Test score (dof = 4) features area and bedrooms: {test_score} %\n')

#Train score (dof = 4) features area and bedrooms: 47.80220051078344 %
#Test score (dof = 4) features area and bedrooms: 52.6614624863633 %


