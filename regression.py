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
