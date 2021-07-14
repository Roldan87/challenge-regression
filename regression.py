
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

