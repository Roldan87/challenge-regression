import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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