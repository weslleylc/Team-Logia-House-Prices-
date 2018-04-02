"""
Created on Tue Mar 20 19:33:25 2018

@author: Weslley Lioba Caldas
"""
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np # linear algebra

class LinearRegression(BaseEstimator, RegressorMixin):  

    def __init__(self, regularization_factor=0):

        self.regularization_factor = regularization_factor

    def fit(self, X, y=None):
        """
        Least Squares Closed Formula Implementation
        """

        assert (type(self.regularization_factor) == float), "The parameter must be float"
       
        X_t=np.transpose(X)
        Lambda=np.identity(len(X_t))*self.regularization_factor
        self.Coefficients=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_t,X)+Lambda),X_t),y)
        
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "Coefficients")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        y_hat=np.matmul(X,self.Coefficients)
            
        return(y_hat)
        
        
