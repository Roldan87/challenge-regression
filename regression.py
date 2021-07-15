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
    linear_model.test_residuals
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


    ######################################## Data preparation #########################################

df = pd.read_csv('jose/df_cut_houses.csv', sep=',', index_col=0)

X = df[['area', 'bedrooms']].values.reshape(-1,2)
Y = df['price']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(20, 500, 100)   # range of area values
y_pred = np.linspace(0, 12, 12)  # range of bedrooms values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

############################################## Evaluate ############################################

r2 = model.score(X, Y)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.3)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Area', fontsize=12)
    ax.set_ylabel('Bedrooms', fontsize=12)
    ax.set_zlabel('Price', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')


ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()

if __name__ == '__main__':
    run()
