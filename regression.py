from lib import dfops as dops
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
    df_ops = dops.DfOps(df_houses, 20, 15)
    df_ops.print_datatypes()


if __name__ == '__main__':
    run()

