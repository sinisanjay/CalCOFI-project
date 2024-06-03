import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import os.path
import time

np.random.seed(42)

#Load Dataset
#import dataset
file_dir = 'D:\Sini-Project\Datasets\CalCOFI'
file1='bottle.csv'
file2 = 'cast.csv'

#dataset = pd.read_csv(file1)
dataset = pd.read_csv(os.path.join(file_dir, file1))

print(dataset.shape)
print(dataset.columns)

#Stratified Split, strata = depth

depth_bins = np.arange(0,1000, 100)
import numpy as np
depth_bins = np.append(depth_bins, [np.inf])
print(depth_bins)

depth_labels = np.arange(1, len(depth_bins))
print('labels=', depth_labels, len(depth_labels))

dataset_copy = dataset.copy()

dataset_copy['depth_cat'] = pd.cut(dataset_copy['Depthm'],
                   bins = depth_bins,
                   labels = depth_labels)

dataset_copy['depth_cat'] = dataset_copy['depth_cat'].fillna(1)

from sklearn.model_selection import StratifiedShuffleSplit
split_obj = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
gen_obj = split_obj.split(dataset_copy, dataset_copy['depth_cat'])

for i, (train_index, test_index) in enumerate(gen_obj):
    print(i,'. ',train_index, ', ', test_index)

print(len(train_index))
print(len(test_index))

strat_train_set = dataset.iloc[train_index]
strat_test_set = dataset.iloc[test_index]

print(strat_train_set.shape)
print(strat_test_set.shape)

print('\n --------------------------------Prepare Data for ML algorithms-----------------------')
train_set = strat_train_set.loc[~strat_train_set['R_TEMP'].isna()]
print(train_set.shape)

test_set = strat_test_set.loc[~strat_test_set['R_TEMP'].isna()]
print(test_set.shape)

drop_attributes = ['Cst_Cnt','Btl_Cnt','Sta_ID', 'Depth_ID', 'BtlNum', 'IncTim', 'DIC Quality Comment']
cat_attributes =['RecInd']

#Separate target variable, temperature and features
X_train = train_set.drop(['T_degC', 'R_TEMP','T_prec','T_qual','R_POTEMP'], axis=1)
X_train = X_train.drop(drop_attributes, axis=1)
y_train = train_set['R_TEMP'].copy()

#Handle missing values
X_train_num = X_train.drop(cat_attributes, axis=1)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

#Categorical
print(X_train['RecInd'].value_counts())
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, list(X_train_num)),
    ('cat', OneHotEncoder(),  ["RecInd"])
])

#Tranform data
X_train_tr = full_pipeline.fit_transform(X_train)
print(X_train_tr[:2, -8:])
print(X_train_tr.shape)
dic = {}
models = []
rmse = []
cross_scores = []
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
start = time.time()

print('\n--------------------Training the model - Linear Regression--------------------')

from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()

#lin_regr.fit(X_train_tr, y_train)
#y_pred = lin_regr.predict(X_train_tr)
#print('label-->', y_train[:5])
#print('Lpred-->', y_pred[:5])
#lin_rmse = mean_squared_error(y_train, y_pred, squared = False)
#print('RMSE=', lin_rmse)
models.append('LinearReg')
#rmse.append(lin_rmse)

lin_scores = -cross_val_score(lin_regr, X_train_tr, y_train,
                         scoring="neg_root_mean_squared_error", cv=5)
cross_scores.append(lin_scores.mean())
print('CROSS VAL RMSE=', lin_scores.mean())

print('\n--------------------Training the model - DecisionTreeRegressor--------------------')
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

#tree_reg.fit(X_train_tr, y_train)
#y_pred = tree_reg.predict(X_train_tr)
#dec_rmse = mean_squared_error(y_train, y_pred, squared = False)
#print(dec_rmse)
models.append('DecisionTree')
#rmse.append(dec_rmse)
dec_scores = -cross_val_score(tree_reg, X_train_tr, y_train,
                         scoring="neg_root_mean_squared_error", cv=5)
print('CROSS VAL RMSE=', dec_scores.mean())
cross_scores.append(dec_scores.mean())

print('\n--------------------Training the model - RandomForestRegressor--------------------')
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
#forest_reg.fit(X_train_tr, y_train)
#y_pred = forest_reg.predict(X_train_tr)
#forest_rmse = mean_squared_error(y_train, y_pred, squared = False)
#print('RMSE=',forest_rmse)
models.append('RandomForest')
#rmse.append(forest_rmse)
rf_scores = -cross_val_score(forest_reg, X_train_tr, y_train,
                         scoring="neg_mean_squared_error", cv=5)
print(rf_scores)
cross_scores.append(rf_scores.mean())
cross_rmse = np.sqrt(rf_scores.mean())
print('CROSS VAL MSE=',rf_scores.mean())

#later added
cross_rmse = np.sqrt(rf_scores.mean())

"""Result = 
MSE= [0.09400711485782162], RMSE = 0.31
time= 8841.667774438858"""

print('\n--------------------Training the model - xgboostRegressor--------------------')
import xgboost
xg_regr = xgboost.XGBRegressor(seed = 42)
#xg_regr.fit(X_train_tr, y_train)
#y_pred = xg_regr.predict(X_train_tr)
#xg_rmse = mean_squared_error(y_train, y_pred, squared = False)
#print(xg_rmse)
models.append('XGBoost')
#rmse.append(xg_rmse)
xg_scores = -cross_val_score(xg_regr, X_train_tr, y_train,
                         scoring="neg_root_mean_squared_error", cv=5)
cross_scores.append(xg_scores.mean())

"""Result = 
RMSE= [0.32585517378145124]
time= 965.3328258991241"""

print(models)
print(rmse)
print(cross_scores)
end = time.time()
print('time in sec=', end-start)