import numpy as np


def sample_boolean_X(n, d):
    X = np.random.randint(0, 2.0, (n, d))
    return X


def sample_normal_X(n, d, mean=0, scale=1, corr=0, Sigma=None):
    if Sigma is not None:
        X = np.random.multivariate_normal(mean, Sigma, size=(n, d))
    elif corr == 0:
        X = np.random.normal(mean, scale, size=(n, d))
    else:
        Sigma = np.zeros((d, d)) + corr
        np.fill_diagonal(Sigma, 1)
        X = np.random.multivariate_normal(mean, Sigma, size=(n, d))
    return X


def sample_normal_X_reorder(support, n, d, mean=0, scale=1, corr=0, Sigma=None):
    X = sample_normal_X(n=n, d=d, mean=mean, scale=scale, corr=corr, Sigma=Sigma)
    for i in range(d):
        if i not in support:
            support.append(i)
    X[:] = X[:, support]
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

