# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
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


# %%
df = pd.read_csv('temp_output.csv', sep=',', index_col=0)


# %%
df.head()


# %%
df.info()


# %%
fig = plt.figure(figsize=(15,15))
plt.style.use('dark_background')
sns.heatmap(df.corr(), annot=True)


# %%
df.price.sort_values(ascending=False).head(10)


# %%
df.bedrooms.sort_values(ascending=False).head(10)


# %%
df_cut = df.copy()


# %%
df_cut = df_cut[df_cut.price <= 5000000.0]


# %%
df_cut = df_cut[df_cut.price >= 10000.0]


# %%
df_cut = df_cut[df_cut.bedrooms <= 10.0]


# %%
df_cut.info()


# %%
x_1 = df_cut['area'].to_numpy().reshape(9978, 1)
x_2 = df_cut['bedrooms'].to_numpy().reshape(9978, 1)
y = df_cut['price']


# %%
print(x_1.shape)
print(x_2.shape)
print(y.shape)
print(type(x_1))
print(type(x_2))
print(type(y))


# %%
def dof_test(dof):
    X_1train, X_1test, y_1train, y_1test = train_test_split(x_1, y, random_state=45, test_size=0.2)
    X_2train, X_2test, y_2train, y_2test = train_test_split(x_2, y, random_state=45, test_size=0.2)
    model_1 = make_pipeline(PolynomialFeatures(degree=dof), linear_model.LinearRegression())
    model_2 = make_pipeline(PolynomialFeatures(degree=dof), linear_model.LinearRegression())
    model_1.fit(X_1train, y_1train)
    model_2.fit(X_2train, y_1train)
    prediction_1 = model_1.predict(X_1test)
    train_score1 = model_1.score(X_1train, y_1train)
    train_score1 = train_score1 * 100
    test_score1 = model_1.score(X_1test, y_1test)
    test_score1 = test_score1 * 100
    print(f'Train score (dof = {dof}) feature area: {train_score1} %')
    print(f'Test score (dof = {dof}) feature area: {test_score1} %')
    prediction_2 = model_2.predict(X_2test)
    train_score2 = model_2.score(X_2train, y_2train)
    train_score2 = train_score2 * 100
    test_score2 = model_2.score(X_2test, y_2test)
    test_score2 = test_score2 * 100
    print(f'Train score (dof = {dof}) feature bedrooms: {train_score2} %')
    print(f'Test score (dof = {dof}) feature bedrooms: {test_score2} %\n')
    return prediction_1, prediction_2


# %%
for dof in range(1,10):
    prediction1, prediction2 = dof_test(dof)


# %%
x = df_cut[['area', 'bedrooms']]
y = df_cut['price']


# %%
def dof_test(dof):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)
    model = make_pipeline(PolynomialFeatures(degree=dof), linear_model.LinearRegression())
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    train_score = train_score * 100
    test_score = model.score(X_test, y_test)
    test_score = test_score * 100
    print(f'Train score (dof = {dof}) features area and bedrooms: {train_score} %')
    print(f'Test score (dof = {dof}) features area and bedrooms: {test_score} %\n')
    return prediction


# %%
for dof in range(1,10):
    multi_prediction = dof_test(dof)


# %%



