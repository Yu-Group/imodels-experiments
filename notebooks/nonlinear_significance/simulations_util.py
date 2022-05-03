import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

def sample_real_X(fpath=None, X=None, seed=None, normalize=True,
                  sample_row_n=None, sample_col_n=None, permute_col=True,
                  signal_features=None, n_signal_features=None, permute_nonsignal_col=None):
    """
    :param fpath: path to X data
    :param X: data matrix
    :param seed: random seed
    :param normalize: boolean; whether or not to normalize columns in data to mean 0 and variance 1
    :param sample_row_n: number of samples to subset; default keeps all rows
    :param sample_col_n: number of features to subset; default keeps all columns
    :param permute_col: boolean; whether or not to permute the columns
    :param signal_features: list of features to use as signal features
    :param n_signal_features: number of signal features; required if permute_nonsignal_col is not None
    :param permute_nonsignal_col: how to permute the nonsignal features; must be one of [None, "block", "indep"], where
        None performs no permutation, "block" performs the permutation row-wise, and "indep" permutes each nonsignal
        feature column independently
    :return:
    """
    assert permute_nonsignal_col in [None, "block", "indep"]
    if X is None:
        X = pd.read_csv(fpath)
    if normalize:
        X = (X-X.mean())/X.std()
    if seed is not None:
        np.random.seed(seed)
    if permute_col:
        X = X[np.random.permutation(X.columns)]
    if sample_row_n is not None:
        X = X.sample(n=sample_row_n, replace=False)#, random_state=1)
    if sample_col_n is not None:
        if signal_features is None:
            X = X.sample(n=sample_col_n, replace=False, axis=1)#, random_state=2)
        else:
            rand_features = np.random.choice([col for col in X.columns if col not in signal_features],
                                             sample_col_n-len(signal_features))
            X = X[signal_features + list(rand_features)]
    if signal_features is not None:
        X = X[signal_features + [col for col in X.columns if col not in signal_features]]
    if permute_nonsignal_col is not None:
        assert n_signal_features is not None
        if permute_nonsignal_col == "block":
            X = np.hstack([X.iloc[:, :n_signal_features].to_numpy(),
                           X.iloc[np.random.permutation(X.shape[0]), n_signal_features:].to_numpy()])
            X = pd.DataFrame(X)
        elif permute_nonsignal_col == "indep":
            for j in range(n_signal_features, X.shape[1]):
                X.iloc[:, j] = np.random.permutation(X.iloc[:, j])

    return X.to_numpy()


def sample_enhancer_X(seed=None, permute=True, sample_frac=1.0,
                      signal_features=None, s=None, permute_cols_indep=True):
    """
    :param seed:
    :param permute: boolean; whether or not to permute columns
    :param sample_frac: sampling fraction for subsetting rows
    :param signal_features: optional character vector specifying true signal features;
        ignored if permute=True
    :param s: optional integer specifying the number of true signal features; if
        provided, all but the first s features get permuted (row-wise)
    :return:
    """
    X = pd.read_csv("data/X_enhancer_uncorrelated_normalized.csv")
    np.random.seed()
    X = X.sample(frac=sample_frac, replace=False, random_state=1)
    if permute == True:
        X = X[np.random.permutation(X.columns)]
    else:
        if signal_features is not None:
            X = X[signal_features + [col for col in X.columns if col not in signal_features]]

    if s is not None:  # then permute non-signal features
        if permute_cols_indep:
            for j in range(s, X.shape[1]):
                X.iloc[:, j] = np.random.permutation(X.iloc[:, j])
        else:
            X = pd.concat([X.iloc[:, :s], X.iloc[:, s:].sample(frac=1.0, replace=False, random_state=10)], axis=1, ignore_index=True)

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


def linear_model(X, sigma, s, beta, heritability = None, snr=None, return_support=False):
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
    if heritability is not None:
        sigma = (np.var(y_train)*((1.0-heritability)/(heritability)))**0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr)**0.5
    y_train = y_train + sigma * np.random.randn((len(X)))
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def sum_of_squares(X, sigma, s, beta, heritability=None, snr=None, return_support=False):
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
    if heritability is not None:
        sigma = (np.var(y_train)*((1.0-heritability)/(heritability)))**0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr)**0.5
    y_train = y_train + sigma * np.random.randn((len(X)))
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def lss_model(X, sigma, m, r, tau, beta, heritability=None, snr=None, return_support=False):
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
    if heritability is not None:
        sigma = (np.var(y_train)*((1.0-heritability)/(heritability)))**0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr)**0.5
    y_train = y_train + sigma * np.random.randn(n)

    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train

def linear_lss_model(X,sigma,m,r,tau,beta, s=None,heritability=None, snr=None, return_support = False):
    """
    This method creates response from an Linear + LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    tau: threshold
    s: sparsity 
    sigma: standard deviation of noise
    beta: coefficient vector. If beta not a vector, then assumed a constant

    :return
    y_train: numpy array of shape (n)
    """
    n,p = X.shape
    if s is None:
        s = m*r
    
    def linear_func(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * beta[j]
        return linear_term
    
    def lss_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        return y
    
    beta_linear = generate_coef(beta, s)
    beta_lss = generate_coef(beta, m)
    
    y_train_linear = np.array([linear_func(X[i, :],s,beta_linear) for i in range(n)])
    y_train_lss = np.array([lss_func(X[i, :], beta_lss) for i in range(n)])
    y_train = np.array([y_train_linear[i]+y_train_lss[i] for i in range(n)])
    if heritability is not None:
        sigma = (np.var(y_train)*((1.0-heritability)/(heritability)))**0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr)**0.5
    y_train = y_train + sigma * np.random.randn(n)
    if return_support:
        support = np.concatenate((np.ones(max(m * r,s)), np.zeros(X.shape[1] - max((m * r),s))))
        return y_train, support, beta_lss
    else:
        return y_train
    
    

def sum_of_polys(X, sigma, m, r, beta, heritability=None, snr=None, return_support=False):
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
    if heritability is not None:
        sigma = (np.var(y_train)*((1.0-heritability)/(heritability)))**0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr)**0.5
    y_train = y_train + sigma * np.random.randn(n)

    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train

