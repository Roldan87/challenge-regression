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
pd.set_option("display.max_columns",None)
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

##################################################
# With polynomial model with Categorique features
##################################################

X_model = df_dummy.iloc[:,:25]
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

###########################################################
# Integrate 'building_state' to the model PolynomialFeatures
###########################################################
#rint('####### df_cat ######')
df_cat = df[['building_state','area','bedrooms','price']]

state_mapping={'to restore':0,'to renovate':1,'to be done up':2,'undefined':3,'as new':4,'just renovated':5,'good':6}
df_cat['building_state'] = df_cat.loc[:,['building_state']].replace(state_mapping)
#print(df_cat)

X_model = df_cat.iloc[:,:2]
y_model = df_cat.iloc[:,[3]]# end column

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,
                                                    test_size=0.20, random_state=0)
print('-----Integrate building_state------')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('-----------')

nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train)

model=make_pipeline(PolynomialFeatures(degree=1),
                   LinearRegression())
model.fit(X_train,y_train)
# model.scores(X_test,y_test)
predictions=model.predict(X_test)
score = r2_score(y_test,y_pred)
print('r2_score (polynomial _ Buiding_state)=', score)

x_p =X_train[:,0]
y_p =y_train[:,0]
#sns.pairplot(data=df_cat)

fig, ax = plt.subplots()
sns.set_style("whitegrid")
sns.lmplot(x=x_p, y=y_p, data=df_cat)
ax.set_xlabel('building_state')
ax.set_ylabel('price')
plt.title("Price/building_state")
plt.legend()
#plt.grid(True)
plt.show()

'''
#######
fig = plt.figure()
plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
x =X[:,0]
y =X[:,1]
z =y
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Y label', rotation=90)
plt.show()
'''

'''
X = pd.DataFrame({'animals':['low','med','low','high','low','high']})
enc = OrdinalEncoder()
enc.fit_transform(X.loc[:,['animals']])

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=['to restore','to renovate','to be done up','undefined','as new','just renovated','good'])
enc.fit_transform(df_cat.loc[:,['building_state']])
print(df_cat)

########################
# Choose features to prepare OneHotEncoding
# area, bedrooms
# region, building_state, property_type, property_subtype
########################
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

#Reorder columns:
df_ohe = df[['region','building_state','property_type','property_subtype','area','bedrooms','price']]
print (df_ohe.head())

X_model = df_ohe.iloc[:,:5]
y_model = df_ohe.iloc[:,[6]]# end column

print(X_model.shape)
print(y_model.shape)

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,test_size=0.20, random_state=0)
print('-----------')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('-----------')

column_trans = make_column_transformer(
    (OneHotEncoder(sparse=False),['region','building_state','property_type','property_subtype']),
    remainder='passthrough')#passthrough

X_train_trans = column_trans.fit_transform(X_train)
X_test_trans = column_trans.fit_transform(X_test)
X_model_trans = column_trans.fit_transform(X_model)


#X_train, X_test, y_train, y_test = train_test_split(X_model_trans, y_model,
#                                                    test_size=0.20, random_state=0)

print(X_train_trans.shape)
print(X_test_trans.shape)
print(X_model_trans.shape)

nb_degree = 1
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_train_trans)

model=make_pipeline(PolynomialFeatures(degree=1),
                   LinearRegression())
model.fit(X_train_trans,y_train)


predictions=model.predict(X_test_trans)
score=model.score(predictions,y_test)
print('Score (OneHotEncoding) =', score) '''
