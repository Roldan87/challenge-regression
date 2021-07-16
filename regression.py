from sklearn.linear_model import LinearRegression
from lib import df_ops as dop
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# todo: numeric state of house
# general rule is to never call fit on the test data
def run():

    df_houses = pd.read_csv('assets/temp_output.csv', sep=',', index_col=0)
    dfo = dop.DfOps(df_houses, 50)
    print(f"Before cleanup: {dfo.df.shape}")

    ################ Clean-up ###################
    # dropping rows for subtype
    column = "property_subtype"
    unwanted_property_types = dfo.strings_in_column_below_threshold_are(column, 0.3)
    dfo.drop_rows_when_column_value_in_list(column, unwanted_property_types)

    # Dropping properties that have more then 12 bedrooms
    dfo.drop_rows_smaller_than_treshold("bedrooms", 12.0)

    # Dropping properties that cost more then 2000000
    dfo.drop_rows_smaller_than_treshold("price", 2000000)

    # Dropping properties with more then 500m2
    dfo.drop_rows_smaller_than_treshold("area", 500.0)

    # reindex after dropping rows
    dfo.reindex()
    print(f"After cleanup: {dfo.df.shape}")
    dfo.write_to_csv("assets/cleaned_data.csv")

    # Area & Bedrooms features vs Price model
    # Features
    features = ["area", "bedrooms"]
    target = "price"
    X = dfo.df[features]
    y = dfo.df[target]

    random_state = 45

    ######################## polynomials ##############################
    train_rmse_errors = []
    train_scores = {}
    test_rmse_errors = []
    test_scores = {}

    for d in range(1, 10):
        poly_converter = PolynomialFeatures(degree=d)
        poly_features = poly_converter.fit_transform(X)
        print(poly_features.shape)
        # print(poly_features[0])  # [1.00000000e+00 1.23000000e+02 2.00000000e+00 1.51290000e+04 2.46000000e+02 4.00000000e+00 1.86086700e+06 3.02580000e+04 4.92000000e+02 8.00000000e+00 2.28886641e+08 3.72173400e+06 6.05160000e+04 9.84000000e+02 1.60000000e+01]

        # split poly features train/test
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, random_state=random_state)

        # fit model on training data
        model = LinearRegression()
        model.fit(X_train, y_train)

        # predict on train & test
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # store/save RMSE for BOTH the train & test set
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        train_rmse_errors.append(train_rmse)
        test_rmse_errors.append(test_rmse)

        score_on_train_set = model.score(X_train, y_train) * 100
        score_on_test_set = model.score(X_test, y_test) * 100

        train_scores[d] = score_on_train_set
        test_scores[d] = score_on_test_set

        print(f"Order: {d}")
        print(f"Train score: {score_on_train_set:.4f}")
        print(f"Test score: {score_on_test_set:.4f}")

    print(train_scores)
    print(test_scores)
    print(train_rmse_errors)
    print(test_rmse_errors)

    # plot the results (RMSE vs poly order)
                # scalex   # scaley
    plt.plot(range(1, 10), train_rmse_errors[:], label="Train RMSE")
    plt.plot(range(1, 10), test_rmse_errors[:], label="Test RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("degree of polynomial")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    # plot train and test score
    plt.clf()
    plt.ylabel("Score")
    plt.xlabel("Degree of Polynomial")
    plt.plot(list(test_scores.values()), label="Test Score")
    plt.plot(list(train_scores.values()), label="Train Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


    ######################## Prepare model data point for visualization ###############################
    print(X.head())
    print("ok")
    x = X["area"]
    Y = X["bedrooms"]  # capital Y as to not interfere with y
    z = y  # target, namely price

    x_pred = np.linspace(20, 500, 100)   # range of area values
    y_pred = np.linspace(0, 12, 12)  # range of bedrooms values
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

    ################################################ Train #############################################
    ols = LinearRegression()
    model = ols.fit(X, Y)
    predicted = model.predict(model_viz)

    ############################################## Evaluate ############################################

    r2 = model.score(X, Y)

    ############################################## Plot ################################################
    plt.clf()
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
    plt.show()

if __name__ == '__main__':
    run()
