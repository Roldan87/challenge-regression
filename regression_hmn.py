import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

def add_cities_provinces_to_dataset(df):

    # df.set_index(df["locality"], inplace=True)
    # df = df.drop(columns='locality')

    data_postcodes = "dataset/postalcode_city_state.csv"

    loc_pc = pd.read_csv(data_postcodes, sep=";")
    # loc_pc = loc_pc.drop(columns='Unnamed: 3')
    loc_pc = loc_pc.drop(columns='Unnamed: 4')
    loc_pc = loc_pc.drop_duplicates(subset=['Postcode'])
    print(loc_pc.Postcode.value_counts())

    df = df.merge(loc_pc, how='inner', left_on='locality', right_on="Postcode")
    df = df.rename(columns={'City': 'city',
                            'State': 'province',
                            'Region':'region'})
    df = df.drop(columns=['Postcode','province'])
    print(df.shape)
    return df

df_origin= pd.read_csv("dataset/temp_output.csv")
pd.set_option("display.max_columns",101)
df = df_origin.copy()
#
# add columns to work regionally
df = add_cities_provinces_to_dataset(df)

print(df.head())
#print(df.info())

#correlations:
correlations = df.corr()
sns.heatmap(correlations)
plt.show()
#print(correlations)

#Observe Price/area:
x = df.area
y = df.price
plt.scatter(x, y, c='c')
plt.title("Observe x / y")
plt.xlabel("area")
plt.ylabel("price")
plt.grid(True)
plt.show()
# we should remove outliers

#Checking missing values ("" or null),
total_na_values = df.isna().sum().sum()
print(total_na_values)# 0: no na in dataframe
#Sum of na values

#control na:
print(f'Sum of na values by columns: \n{df.isna().sum()} \n')

#Drop duplicates
# a. Check for unique of data (cols)
df.duplicated()
print("Number Of Rows In The Original DataFrame:", len(df))
print("Number Of Rows After Dedupling:", len(df.drop_duplicates()))
#No duplicates

#Observe values in each column (unique):
get_column_names = df.columns
for i in range(len(get_column_names)):
    print(f'---Count values unique from column * {df.columns[i]} * ---: \n {df[df.columns[i]].unique()} \n')

#Filter outliers:
df = df[df.area <= 500.0]
df = df[df.price <= 1000000]
df = df[df.bedrooms < 10.0]
df.shape

#####################
# Test feature "area"
#####################
X_model = df[['area']].values
y_model = df[['price']].values

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.20, random_state=0)
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=1),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
plt.scatter(X_train, y_train, c='c', label='data')
plt.scatter(X_test, predictions, c='r', label='Prediction')
plt.title("Price/areas : Observe X_test / predictions")
plt.xlabel("areas")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
score=model.score(X_test,y_test)
print('Score (area)=', score)

#########################
# Test feature "bedrooms"
#########################

X_model = df[['bedrooms']]
y_model = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.20, random_state=0)

nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=1),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
plt.scatter(X_train, y_train, c='c', label='data')
plt.scatter(X_test, predictions, c='r', label='Prediction')
plt.title("Price/numbers bedrooms : Observe X_test / predictions")
plt.xlabel("bedrooms")
plt.ylabel("price")
plt.grid(True)
plt.legend()
plt.show()
score=model.score(X_test,y_test)
print('Score (bedrooms) =', score)

####################################
# Test features "area" and "bedrooms"
####################################
X_model = df[['area','bedrooms']]
y_model = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.20, random_state=0)
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=4),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
score=model.score(X_test,y_test)
print('Score (area+bedrooms) =', score) # 0.3193

##############
# With method dummies_adress
##############
df_dummy = df.drop(columns=['facades','kitchen_equipped','furnished',
                            'open_fire','locality','land_surface',
                            'terrace','terrace_surface','swimming_pool',
                            'city','garden',
                            'garden_surface'])
#print(df_dummy.head())
#Reorder columns:
df_dummy = df_dummy[['region','building_state','property_type','property_subtype','area','bedrooms','price']]

# Convert 'region'
df1=pd.get_dummies(df_dummy['region'],drop_first=True)
df_dummy = pd.concat([df1,df_dummy], axis=1)
df_dummy.drop('region', axis=1, inplace=True)
#print(df_dummy)

# Convert 'building_state'
df1=pd.get_dummies(df_dummy['building_state'],drop_first=True)
df_dummy = pd.concat([df1,df_dummy], axis=1)
df_dummy.drop('building_state', axis=1, inplace=True)
#print(df_dummy)

# Convert 'property_type'
df1=pd.get_dummies(df_dummy['property_type'],drop_first=True)
df_dummy = pd.concat([df1,df_dummy], axis=1)
df_dummy.drop('property_type', axis=1, inplace=True)
#print(df_dummy)

# Convert 'property_subtype'
df1=pd.get_dummies(df_dummy['property_subtype'],drop_first=True)
df_dummy = pd.concat([df1,df_dummy], axis=1)
df_dummy.drop('property_subtype', axis=1, inplace=True)
#print(df_dummy)


#Linear model
X_model = df_dummy.iloc[:,:-1]
y_model = df_dummy.iloc[:,[26]] # end column

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,
                                                    test_size=0.20, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Score X/y_train_80% =',regressor.score(X_train, y_train))
print('Score X/y_model_100% =',regressor.score(X_model, y_model))

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print('r2_score (linear) =', score)

#######################
# With polynomial model
#######################

X_model = df_dummy.iloc[:,:-1]
y_model = df_dummy.iloc[:,[26]]# end column

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,
                                                    test_size=0.20, random_state=0)

nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=2),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
score = r2_score(y_test,y_pred)
print('r2_score (polynomial)=', score)

'''
########################
# Choose features to prepare OneHotEncoding
# area, bedrooms
# region, building_state, property_type, property_subtype
########################
from sklearn.preprocessing import OneHotEncoder

X_model = df_dummy.iloc[:,:-1]
y_model = df_dummy.iloc[:,[26]]# end column

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,
                                                    test_size=0.20, random_state=0)
ohe = OneHotEncoder()


dfle = df.copy()
dfle.region=le.fit_transform(dfle.region)
dfle.building_state = le.fit_transform(dfle.building_state)
dfle.property_type=le.fit_transform(dfle.property_type)
dfle.property_subtype=le.fit_transform(dfle.property_subtype)
#print(dfle.head())
y_model = dfle[['price']].values
X_model = dfle[['region','building_state','property_type','property_subtype','area','bedrooms']].values

#print(X_model)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories='auto')
#ohe = OneHotEncoder(categorical_features=[0])
X_model = ohe.fit_transform(X_model).toarray()

df_model = pd.DataFrame(X_model)
print('df_model =')
print(df_model.head())

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.20, random_state=0)


nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=1),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
score=model.score(X_test,y_test)
print('Score (OneHotEncoding) =', score)
'''