# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:33:25 2018

@author: Weslley Lioba Caldas
"""

import utils
import regression
import pandas as pd # read data CSV file
import numpy as np # linear algebra
from scipy.stats import skew #for some statistics
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
import lightgbm as lgb
import matplotlib.pyplot as plt  # Matlab-style plotting
from IPython import get_ipython
#make the plot in line
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


#read the data
data_train  = pd.read_csv('input/train.csv')
data_test  = pd.read_csv('input/test.csv')

#---------------------------Target Variable----------------------------

#analisyng the target variable
#descriptive statistics summary
data_train['SalePrice'].describe()

#If the Data(in this case, the response variable) follows a Normal distribution, the errors of this model should have the same variance. By the way, the coefficients are more easy to be interpretable.

#sns.distplot(data_train['SalePrice'], fit=stats.norm)

# histogram , Probplot, skewness and kurtosis to check the shape of distribution
utils.analitycalPlot(data_train,'SalePrice')

# the data deviates from the normal distribution and is right skewed. 
#The data not follows a Normal Distribution, but its possible use data transformations functions like logaritica, wuadradict and exponential to avoid this problem. 

#The log transformation can be used to make highly skewed distributions less skewed. 
original_label= data_train['SalePrice']

data_train['SalePrice'] = np.log1p(data_train['SalePrice'])
#now, we check again
utils.analitycalPlot(data_train,'SalePrice')

#---------------------------Data Analisys----------------------------------




#---------------------------Correlation ----------------------------------

#Correlation betwwen the features and SalePrice
corrmat = data_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)


#---------------------------Remove Outliars ----------------------------------

#I am select the most influency variables
FontSize=13;

utils.simplePlot(data_train,'GrLivArea','SalePrice','GrLivArea','SalePrice',FontSize)

#plot for OverallQual 
utils.simplePlot(data_train,'OverallQual','SalePrice','OverallQual','SalePrice',FontSize)

#plot for TotalBsmtSF
utils.simplePlot(data_train,'TotalBsmtSF','SalePrice','TotalBsmtSF','SalePrice',FontSize)

#plot for YearBuilt
utils.simplePlot(data_train,'YearBuilt','SalePrice','YearBuilt','SalePrice',FontSize)

#Deleting outliers
data_train.drop(data_train[(data_train['GrLivArea']>4000) & (data_train['SalePrice']<12.5)].index, inplace=True)

data_train.drop(data_train[(data_train['TotalBsmtSF']>5000) & (data_train['SalePrice']<12.5)].index, inplace=True)


#---------------------------Treatment of  Data ----------------------------------
#drop id feature, because its usefull
len_train=len(data_train)
train_id=data_train['Id']
data_train.drop("Id", axis = 1, inplace = True)

len_test=len(data_test)
test_id=data_test['Id']
data_test.drop("Id", axis = 1, inplace = True)

#concatenate the data for make the processo to clean up more easy
all_data = pd.concat((data_train, data_test)).reset_index(drop=True)
train_label=data_train['SalePrice']
all_data.drop(['SalePrice'], axis=1, inplace=True)



#---------------------------Missing Data ----------------------------------

#First we analized the missing data
all_missing=all_data.isnull().sum() / len(all_data)
missing_data = pd.concat([all_data.isnull().sum(),all_missing], axis=1, keys=['Total of missing values', 'Percentage'])
#10 more missing values
all_missing.sort_values(ascending=False)[:10]


figure, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percentage'].values)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=18)
plt.title('Percent missing data by feature', fontsize=18)



#---------------------------Input Data ----------------------------------
# we have two common ways to trat missing data.
#for a small percent of missing data we can using methods to input the missing values
#for a big percent of missing data we just drop the features

#---------------------------case 1---------------------------------------
#these features has 50% or more of missing data, so we drop them.
all_data.drop(['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1, inplace=True)

#---------------------------case 2---------------------------------------
#we fill these missing values using the mean of Linear feet of street of the Neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#---------------------------case 3---------------------------------------
#Utilities : this its a unsefull features since basicaly all instances are "AllPub"
all_data = all_data.drop(['Utilities'], axis=1)

#---------------------------case 4---------------------------------------
# these features has a litle bit of missing data instances, so we just replace the missing values for the most frequent category
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#---------------------------case 5---------------------------------------
#most part of these features has missing values, that probabily represent the absence of something(eg basement,GarageArea). We replace 0 for numerical and 'none' for categorical features
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
#all_data = all_data.drop(['GarageYrBlt','GarageArea'], axis=1)
 
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)    
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')        

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#---------------------------case 5---------------------------------------
#from the data descritption "Assume typical unless deductions are warranted"
all_data["Functional"] = all_data["Functional"].fillna("Type")


#---------------------------Transforming Data ----------------------------------
#some class has numbers but are categorical 
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#---------------------------Label Enconding ----------------------------------

cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# one-hot econding
for column in cols:
    enconder = LabelEncoder() 
    enconder.fit(list(all_data[column].values)) 
    all_data[column] = enconder.transform(list(all_data[column].values))


#---------------------------Create Features ----------------------------------
#Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#we dont need these features anymore
all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


#---------------------------Trating Skewed Features ----------------------------------
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#0.5 its normal distribution, if the data are skewed we aply log1 transformation
lam = 0.15
for i in numeric_feats:
    if(abs(skew(all_data[i].values))>0.75):
        all_data[i] = np.log1p(all_data[i])

#---------------------------Split Data ----------------------------------

all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:len_train]
test = all_data[len_train:]

#---------------------------Model ----------------------------------

#Validation function

n_folds = 5
    
random_grid = {'alpha': [0.0001,0.0001,0.001,0.01,0.1,1.0,1.1]}

lasso=utils.cros_validation(Lasso(random_state=1),train.values,train_label,n_folds,random_grid)

#My implemantation of l2 regularization
random_grid = {'regularization_factor': [0.0001,0.0001,0.001,0.01,0.1,1.0,1.1]}

multi=utils.cros_validation(regression.LinearRegression(regularization_factor=1.0),train.values,train_label,n_folds,random_grid)

#Univariate Case
X=[train['GrLivArea'].values , np.ones(len(train))]
X=np.transpose(X)
uni=utils.cros_validation(regression.LinearRegression(regularization_factor=1.0),X,train_label,n_folds,random_grid)

random_grid = {'n_estimators': [100,200,300,400,500,600,700,720,740,760,780,800]}


lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=800,
                              max_bin = 60, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb=utils.cros_validation(lgb,train.values,train_label,n_folds,random_grid)


#call k-folds validation 
utils.scores('Lasso',utils.cv_rmse(lasso,train.values,train_label,n_folds))


utils.scores('Multivariate Linear Regression',utils.cv_rmse(multi,train.values,train_label,n_folds))


utils.scores('Univariate Linear Regression',utils.cv_rmse(uni,X,train_label,n_folds))

utils.scores('Gradient Boosting',utils.cv_rmse(model_lgb,train.values,train_label,n_folds))


#---------------------------Meta Learning ----------------------------------

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = np.exp(lasso.predict(test)*0.2+multi.predict(test)*0.2+model_lgb.predict(test)*0.6)
sub.to_csv('submission.csv',index=False)



#---------------------------Model implemantation withou schilearning ----------------------------------
"""
#Least Squares Closed Formula (univariate)

X=[train['GrLivArea'].values , np.ones(len(train))]
X_t=X
X=np.transpose(X)
y=train_label.values
B=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_t,X)),X_t),y)

y_hat=np.matmul(X,B)

#Least Squares Closed Formula (multivariate)

X=[train.values , np.ones(len(train))]
X_t=np.transpose(X)
y=train_label.values
B=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_t,X)),X_t),y)

y_hat=np.matmul(X,B)
""" 
 #---------------------------End ----------------------------------


