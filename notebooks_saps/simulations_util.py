import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from imodels import SaplingSumRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def sample_uniform_X(n,d):
    X = np.random.uniform(0,1.0,(n,d))
    return X

def sample_boolean_X(n,d):
    X = np.random.randint(0,2.0,(n,d))
    return X

def linear_model(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed that 
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i]*beta
        return linear_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train

def sum_of_squares(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed that 
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i]*x[i]*beta
        return linear_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train

def CART(X_train,y_train,X_saps,y_saps,X_test,y_test, saps = False):
    '''Note X_saps / y_saps are currently ignored
    '''
    if not saps:
        CART = DecisionTreeRegressor(min_samples_leaf = 5)
        CART.fit(X_train,y_train)
        CART_preds = CART.predict(X_test)
        return mean_squared_error(CART_preds,y_test)
    else:
        saps = SaplingSumRegressor()
        saps.fit(X_train,y_train)
        SAPS_preds = saps.predict(X_test)
        return mean_squared_error(SAPS_preds, y_test)
        
def train_all_models(X_train,y_train,X_saps,y_saps,X_test,y_test):
    saps_CART =  CART(X_train,y_train,X_saps,y_saps,X_test,y_test,saps = True)
    cart = CART(X_train,y_train,X_saps,y_saps,X_test,y_test,saps = False)
    #saps_CART_CCP,dissaps_CART_CCP = CART_CCP(X_train,y_train,X_saps,y_saps,X_test,y_test,sigma,k = 5)
    return saps_CART, cart

def log_list(t):
    return [log(x,math.e) for x in t]

def get_best_fit_line(x,y):
    m, b = np.polyfit(x, y, 1)
    return [m,b]


"""
def CART_CCP(X_train,y_train,X_saps,y_saps,X_test,y_test,sigma,k = 5):
    id_threshold = sigma**2/len(X_train)
    alphas = np.geomspace(0.1*id_threshold, 1000*id_threshold, num=5)
    scores = []
    models = []
    for alpha in alphas:
        CART = DecisionTreeRegressor(min_samples_leaf = 5,ccp_alpha = alpha)
        CART.fit(X_train,y_train)
        models.append(CART)
        scores.append(cross_val_score(CART, X_train, y_train, cv=k).mean())
        best_CART = models[scores.index(max(scores))]
        dissaps_MSE = mean_squared_error(best_CART.predict(X_test),y_test)
        saps_MSE = 0 #get_saps_test_MSE(best_CART,X_saps,y_saps,X_test,y_test)
        return saps_MSE,d issaps_MSE
"""