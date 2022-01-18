import numpy as np


def sample_boolean_X(n, d):
    X = np.random.randint(0, 2.0, (n, d))
    return X


def linear_model(X, s, beta, sigma):
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

    def create_y(x, s, beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i] * beta
        return linear_term

    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train


def sum_of_squares(X, s, beta, sigma):
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

    def create_y(x, s, beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i] * x[i] * beta
        return linear_term

    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train


def get_best_fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    return [m, b]
