"""
Created on Tue Mar 20 19:33:25 2018

@author: Weslley Lioba Caldas
"""
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt  # Matlab-style plotting
import pandas as pd
from IPython import get_ipython
#make the plot in line
get_ipython().run_line_magic('matplotlib', 'inline')


def simplePlot(data,xlabel,ylabel,x,y,font_size):
    figure, axis = plt.subplots()
    axis.scatter(x = data[x], y = data[y])
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.show()
    
def simplePlot2(data,xlabel,ylabel,data_x,data_y,font_size):
    figure, axis = plt.subplots()
    axis.scatter(x = data_x, y = data_y)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.show()    
    
def analitycalPlot(data,variable):
    #histogram to check the shape of distribution
    sns.distplot(data[variable], fit=stats.norm)
    # Probplot
    plt.figure()
    stats.probplot(data[variable], plot=plt)
    plt.show()
    #skewness and kurtosis for this data
    print("Skewness: %f" % data[variable].skew())
    print("Kurtosis: %f" % data[variable].kurt())    

def boxPlot(data,variable,variable2):
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=variable, y=variable2, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    plt.show()
    
def scores(variable,score):
    print("\n",variable,"score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

def cv_rmse(model, x,y,nFolds):
    kf = KFold(nFolds, shuffle=True).get_n_splits(x)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error",cv = kf))
    return(rmse)
   
def cros_validation(model, X_train,y_train,nFolds,tuned_parameters):  
    clf = GridSearchCV(model, tuned_parameters, cv=nFolds,scoring="neg_mean_squared_error")
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def rmse(y_hat,y):
    return np.sqrt(((y_hat - y) ** 2).mean())

def model_eval(model,X,y):
    model_fit = model.fit(X, y)
    R2 = cross_val_score(model_fit, X, y, cv=10 , scoring='r2').mean()
    RMSE = np.sqrt(-cross_val_score(model, X, y, cv=10 , scoring='neg_mean_squared_error'))
    print('R2 Score:', R2, '|', 'RMSE:', RMSE.mean())