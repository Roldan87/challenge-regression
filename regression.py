
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
    # df_houses = pd.read_csv("final_list_houses_dataset.csv", sep=',').iloc[:, 1:]
    df_houses = pd.read_csv('assets/temp_output.csv', sep=',', index_col=0)
    dfo = dop.DfOps(df_houses, 50)
    # df.print_datatypes()

    """
    ['good'
     'just renovated'
     'as new'
     'to renovate'
     'undefined'
     'to be done up'
     'to restore']
    """
    # building_state
    # property_subtype
    ['house'
     'villa'
     'mixed'
     'town'
     'farmhouse'
     'chalet'
     'country'
     'exceptional'
     'building'
     'apartment'
     'mansion'
     'bungalow'
     'other'
     'manor'
     'castle'
     'land']

    unwanted_property = ['other', 'manor', 'castle', 'land', 'chalet']
    column = "property_subtype"

    print(dfo.count_rows_having_strings_in_column(column, unwanted_property))  # 70
    print(dfo.count_rows_having_strings_in_column2(column, unwanted_property))
    print(dfo.df.shape[0])  # row count 10058
    dfo.drop_rows_having_strings_in_column(column, unwanted_property)
    print(dfo.df.shape[0])  # row count 9988, rows have been dropped

if __name__ == '__main__':
    run()

<<<<<<< HEAD
# CODE LEONOR

df = pd.read_csv("temp_output (1).csv")

#copy from data frame
df_copy = df.copy()

#dropping column property_type to reduce redundancy
df_copy.drop(columns="property_type")

#Dropping properties that have more then 12 bedrooms
df_copy.bedrooms.value_counts()
df_copy['bedrooms'] = df_copy[df_copy['bedrooms'] < 12.00000]

#Dropping properties that cost more then 2000000
print("there are", df_copy[df_copy['price'] >2000000].value_counts().sum(), "properties in this data set that cost more then 2000000€")
df_copy['price'] = df_copy[df_copy['price'] < 2000000]

#Dropping properties with more then 500m2
df_copy['area'] = df_copy[df_copy['area'] < 500.000000]
df_copy.area.value_counts()
=======





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


>>>>>>> 9e92d4598e17cd288ee9ec59e8c09a0f55d75ebf
