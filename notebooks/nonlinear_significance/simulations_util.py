import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
def sample_enhancer_X(seed = None,permute = True):
    X = pd.read_csv("data/X_enhancer_uncorrelated.csv")
    np.random.seed()
    if permute == True:
        return X.sample(frac=1, axis=1).to_numpy()
    else: 
        return X.to_numpy()

def sample_boolean_X(n, d):
    """
    Sample X with iid boolean entries
    :param n:
    :param d:
    :return:
    """
    X = np.random.randint(0, 2.0, (n, d))
    return X


def sample_normal_X(n, d, mean=0, scale=1, corr=0, Sigma=None):
    """
    Sample X with iid normal entries
    :param n:
    :param d:
    :param mean:
    :param scale:
    :param corr:
    :param Sigma:
    :return:
    """
    if Sigma is not None:
        if np.isscalar(mean):
            mean = np.repeat(mean, d)
        X = np.random.multivariate_normal(mean, Sigma, size=n)
    elif corr == 0:
        X = np.random.normal(mean, scale, size=(n, d))
    else:
        Sigma = np.zeros((d, d)) + corr
        np.fill_diagonal(Sigma, 1)
        if np.isscalar(mean):
            mean = np.repeat(mean, d)
        X = np.random.multivariate_normal(mean, Sigma, size=n)
    return X


def sample_X(support, X_fun, **kwargs):
    """
    Wrapper around dgp function for X that reorders columns so support features are in front
    :param support:
    :param X_fun:
    :param kwargs:
    :return:
    """
    X = X_fun(**kwargs)
    for i in range(X.shape[1]):
        if i not in support:
            support.append(i)
    X[:] = X[:, support]
    return X


def sample_ar1_X(n, d, rho, mean=0):
    """
    Sample X from N(mean, Sigma) where Sigma is an AR1(rho) covariance matrix
    :param n:
    :param d:
    :param rho:
    :param mean:
    :return:
    """
    col1 = [rho**i for i in range(d)]
    Sigma = toeplitz(c=col1)
    X = sample_normal_X(n=n, d=d, mean=mean, Sigma=Sigma)
    return X


def sample_block_cor_X(n, d, rho, n_blocks, mean=0):
    """
    Sample X from N(mean, Sigma) where Sigma is a block diagnoal covariance matrix
    :param n:
    :param d:
    :param rho:
    :param n_blocks:
    :param mean:
    :return:
    """
    Sigma = np.zeros((d, d))
    block_size = d // n_blocks
    if np.isscalar(rho):
        rho = np.repeat(rho, n_blocks)
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        if i == (n_blocks - 1):
            end = d
        Sigma[start:end, start:end] = rho[i]
    np.fill_diagonal(Sigma, 1)
    X = sample_normal_X(n=n, d=d, mean=mean, Sigma=Sigma)
    return X


def generate_coef(beta, s):
    if isinstance(beta, int) or isinstance(beta, float):
        beta = np.repeat(beta, repeats=s)
    return beta


def linear_model(X, sigma, s, beta, return_support=False):
    '''
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''

    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * beta[j]
        return linear_term

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def sum_of_squares(X, sigma, s, beta, return_support=False):
    '''
    This method is used to create responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''

    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * x[j] * beta[j]
        return linear_term

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def lss_model(X, sigma, m, r, tau, beta, return_support=False):
    """
    This method creates response from an LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    tau: threshold
    sigma: standard deviation of noise
    beta: coefficient vector. If beta not a vector, then assumed a constant

    :return
    y_train: numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r  # Cannot have more interactions * size than the dimension

    def lss_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        return y

    beta = generate_coef(beta, m)
    y_train = np.array([lss_func(X[i, :], beta) for i in range(n)])
    y_train = y_train + sigma * np.random.randn(n)

    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train


def sum_of_polys(X, sigma, m, r, beta, return_support=False):
    """
    This method creates response from an LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    sigma: standard deviation of noise
    beta: coefficient vector. If beta not a vector, then assumed a constant

    :return
    y_train: numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r  # Cannot have more interactions * size than the dimension

    def poly_func(x, beta):
        y = 0
        for j in range(m):
            poly_term_components = x[j * r:j * r + r]
            poly_term = np.prod(poly_term_components)
            y += poly_term * beta[j]
        return y

    beta = generate_coef(beta, m)
    y_train = np.array([poly_func(X[i, :], beta) for i in range(n)])
    y_train = y_train + sigma * np.random.randn(n)

    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train

