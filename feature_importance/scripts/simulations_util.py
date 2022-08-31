import numpy as np
import pandas as pd
import random
from scipy.linalg import toeplitz
import warnings


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
    :param permute_nonsignal_col: how to permute the nonsignal features; must be one of
        [None, "block", "indep", "augment"], where None performs no permutation, "block" performs the permutation
        row-wise, "indep" permutes each nonsignal feature column independently, "augment" augments the signal features
        with the row-permuted X matrix.
    :return:
    """
    assert permute_nonsignal_col in [None, "block", "indep", "augment"]
    if X is None:
        X = pd.read_csv(fpath)
    if normalize:
        X = (X - X.mean()) / X.std()
    if seed is not None:
        np.random.seed(seed)
    if permute_col:
        X = X[np.random.permutation(X.columns)]
    if sample_row_n is not None:
        keep_idx = np.random.choice(X.shape[0], sample_row_n, replace=False)
        X = X.iloc[keep_idx, :]
        # X = X.sample(n=sample_row_n, replace=False)#, random_state=1)
    if sample_col_n is not None:
        if signal_features is None:
            X = X.sample(n=sample_col_n, replace=False, axis=1)  # , random_state=2)
        else:
            rand_features = np.random.choice([col for col in X.columns if col not in signal_features],
                                             sample_col_n - len(signal_features), replace=False)
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
        elif permute_nonsignal_col == "augment":
            X = np.hstack([X.iloc[:, :n_signal_features].to_numpy(),
                           X.iloc[np.random.permutation(X.shape[0]), :].to_numpy()])
            X = IndexedArray(pd.DataFrame(X).to_numpy(), index=keep_idx)
            return X
    return X.to_numpy()


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


def sample_block_cor_X(n, d, rho, n_blocks, mean=0):
    """
    Sample X from N(mean, Sigma) where Sigma is a block diagonal covariance matrix
    :param n: number of samples
    :param d: number of features
    :param rho: correlation or vector of correlations
    :param n_blocks: number of blocks
    :param mean: mean of normal distribution
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
    
    
def sample_boolean_X(n, d):
    """
    Sample X with iid boolean entries
    :param n:
    :param d:
    :return:
    """
    X = np.random.randint(0, 2.0, (n, d))
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


def generate_coef(beta, s):
    if isinstance(beta, int) or isinstance(beta, float):
        beta = np.repeat(beta, repeats=s)
    return beta


def corrupt_leverage(X_support, y_train, frac_corrupt, corrupt_quantile):
    ranked_rows = np.apply_along_axis(np.linalg.norm, axis=1, arr=X_support).argsort().argsort()
    low_idx = np.where(ranked_rows < round(corrupt_quantile * len(y_train)))[0]
    hi_idx = np.where(ranked_rows >= (len(y_train) - round(corrupt_quantile * len(y_train))))[0]
    low_switch = np.random.choice(low_idx, size=round(frac_corrupt * len(low_idx)), replace=False)
    hi_switch = np.random.choice(hi_idx, size=round(frac_corrupt * len(hi_idx)), replace=False)
    y_low = y_train[low_switch]
    y_hi = y_train[hi_switch]
    y_train[hi_switch] = y_low
    y_train[low_switch] = y_hi
    return y_train


def linear_model(X, sigma, s, beta, heritability=None, snr=None, error_fun=None,
                 frac_corrupt=None, corrupt_how='permute', corrupt_quantile=None, return_support=False):
    """
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """
    n, p = X.shape
    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * beta[j]
        return linear_term

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    if frac_corrupt is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage":
            y_train = corrupt_leverage(X[:, :s], y_train, frac_corrupt, corrupt_quantile)
            y_train = y_train + sigma * error_fun(n)

    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def logistic_model(X, s, beta,return_support=False):
    """
    This method is used to create responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """

    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * beta[j]
        prob = 1 / (1 + np.exp(-linear_term))
        return (np.random.uniform(size=1) < prob) * 1

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))]).ravel()
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train
    

def logistic_lss_model(X, sigma, m, r, tau, beta, min_active=None, return_support = False):
    """
    This method is used to create responses from a logistic model model with lss
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """
    n, p = X.shape

    def lss_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        prob = 1 / (1 + np.exp(-y))
        return (np.random.uniform(size=1) < prob) * 1

    def lss_vector_fun(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        max_iter = 100
        features = np.arange(p)
        support_idx = []
        for j in range(m):
            cnt = 0
            while True:
                int_features = np.random.choice(features, size=r, replace=False)
                lss_term_components = x_bool[:, int_features]
                lss_term = np.apply_along_axis(all, 1, lss_term_components)
                cnt += 1
                if np.mean(lss_term) >= min_active or cnt > max_iter:
                    y += lss_term * beta[j]
                    features = list(set(features).difference(set(int_features)))
                    support_idx.append(int_features)
                    if cnt > max_iter:
                        warnings.warn("Could not find interaction {} with min active >= {}".format(j, min_active))
                    break
        prob = 1 / (1 + np.exp(-y))
        y = (np.random.uniform(size=n) < prob) * 1
        support_idx = np.stack(support_idx).ravel()
        support = np.zeros(p)
        for j in support_idx:
            support[j] = 1
        return y, support

    beta = generate_coef(beta, m)
    if tau == 'median':
        tau = np.median(X,axis = 0)
    
    if min_active is None:
        y_train = np.array([lss_func(X[i, :], beta) for i in range(n)]).ravel()
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
    else:
        y_train, support = lss_vector_fun(X, beta)
        y_train = y_train.ravel()

    if return_support:
        return y_train, support, beta
    else:
        return y_train

def logistic_partial_linear_lss_model(X,s, m, r, tau, beta, min_active=None, return_support = False):
    """
    This method is used to create responses from a logistic model model with lss
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r
    
    def partial_linear_func(x,s,beta):
        y = 0.0
        count = 0
        for j in range(m):
            for i in range(s):
                y += beta[count]*x[j*r+i]
                count += 1
        return y
    
    def lss_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        return y
    def logistic_link_func(y):
        prob = 1 / (1 + np.exp(-y))
        return (np.random.uniform(size=1) < prob) * 1

    def lss_vector_fun(x, beta, beta_linear):
        x_bool = (x - tau) > 0
        y = 0
        max_iter = 100
        features = np.arange(p)
        support_idx = []
        for j in range(m):
            cnt = 0
            while True:
                int_features = np.concatenate(
                    [np.arange(j*r, j*r+s), np.random.choice(features, size=r-s, replace=False)]
                )
                lss_term_components = x_bool[:, int_features]
                lss_term = np.apply_along_axis(all, 1, lss_term_components)
                cnt += 1
                if np.mean(lss_term) >= min_active or cnt > max_iter:
                    norm_constant = sum(np.var(x[:, (j*r):(j*r+s)], axis=0) * beta_linear[(j*s):((j+1)*s)]**2)
                    relative_beta = beta[j] / sum(beta_linear[(j*s):((j+1)*s)])
                    y += lss_term * relative_beta * np.sqrt(norm_constant) / np.std(lss_term)
                    features = list(set(features).difference(set(int_features)))
                    support_idx.append(int_features)
                    if cnt > max_iter:
                        warnings.warn("Could not find interaction {} with min active >= {}".format(j, min_active))
                    break
        support_idx = np.stack(support_idx).ravel()
        support = np.zeros(p)
        for j in support_idx:
            support[j] = 1
        return y, support    

    beta_lss = generate_coef(beta, m)
    beta_linear = generate_coef(beta, s*m)
    if tau == 'median':
        tau = np.median(X,axis = 0)
    y_train_linear = np.array([partial_linear_func(X[i, :],s,beta_linear ) for i in range(n)])
    if min_active is None:
        y_train_lss = np.array([lss_func(X[i, :], beta_lss) for i in range(n)])
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
    else:
        y_train_lss, support = lss_vector_fun(X, beta_lss, beta_linear)
    y_train = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])
    y_train = np.array([logistic_link_func(y_train[i]) for i in range(n)])
    y_train = y_train.ravel()
    
    if return_support:
        return y_train, support, beta
    else:
        return y_train
    
    
    
def logistic_hier_model(X, sigma, m, r, tau, beta,return_support = False):
    
    n, p = X.shape
    assert p >= m * r

    def reg_func(x, beta):
        y = 0
        for i in range(m):
            hier_term = 1.0
            for j in range(r):
                hier_term += x[i * r + j] * hier_term
            y += hier_term * beta[i]
        return y
    
    def logistic_link_func(y):
        prob = 1 / (1 + np.exp(-y))
        return (np.random.uniform(size=1) < prob) * 1

    beta = generate_coef(beta, m)
    y_train = np.array([reg_func(X[i, :], beta) for i in range(n)])
    y_train = np.array([logistic_link_func(y_train[i]) for i in range(n)])
    y_train = y_train.ravel()
    
    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train
    
def logistic_sum_of_poly(X,m,r,beta,return_support = False):
    n, p = X.shape
    assert p >= m * r

    def reg_func(x, beta):
        y = 0
        for j in range(m):
            poly_term_components = x[j * r:j * r + r]
            poly_term = np.prod(poly_term_components)
            y += poly_term * beta[j]
        return y
       
    def logistic_link_func(y):
        prob = 1 / (1 + np.exp(-y))
        return (np.random.uniform(size=1) < prob) * 1

    beta = generate_coef(beta, m)
    y_train = np.array([reg_func(X[i, :], beta) for i in range(n)])
    y_train = np.array([logistic_link_func(y_train[i]) for i in range(n)])
    y_train = y_train.ravel()
    
    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train
    
    
def sum_of_polys(X, sigma, m, r, beta, heritability=None, snr=None, error_fun=None,
                 frac_corrupt = 0.0,return_support=False):
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
    if error_fun is None:
        error_fun = np.random.randn
    num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
    corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
    non_corrupt_indices = list(set([*range(len(y_train))]) - set(corrupt_indices))
    y_train[corrupt_indices] = y_train[corrupt_indices] + sigma*np.random.standard_cauchy(size=len(corrupt_indices))
    y_train[non_corrupt_indices] = y_train[non_corrupt_indices] + sigma*error_fun((len(non_corrupt_indices)))
    #y_train = y_train + sigma * error_fun(n)

    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train


def sum_of_squares(X, sigma, s, beta, heritability=None, snr=None, error_fun=None, return_support=False):
    """
    This method is used to create responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    sigma: s.d. of added noise
    Returns:
    numpy array of shape (n)
    """

    def create_y(x, s, beta):
        linear_term = 0
        for j in range(s):
            linear_term += x[j] * x[j] * beta[j]
        return linear_term

    beta = generate_coef(beta, s)
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    y_train = y_train + sigma * error_fun((len(X)))
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def lss_model(X, sigma, m, r, tau, beta, heritability=None, snr=None, error_fun=None, min_active=None,
              frac_corrupt=None, corrupt_how='permute', corrupt_quantile=None, return_support=False):
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

    def lss_vector_fun(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        max_iter = 100
        features = np.arange(p)
        support_idx = []
        for j in range(m):
            cnt = 0
            while True:
                int_features = np.random.choice(features, size=r, replace=False)
                lss_term_components = x_bool[:, int_features]
                lss_term = np.apply_along_axis(all, 1, lss_term_components)
                cnt += 1
                if np.mean(lss_term) >= min_active or cnt > max_iter:
                    y += lss_term * beta[j]
                    features = list(set(features).difference(set(int_features)))
                    support_idx.append(int_features)
                    if cnt > max_iter:
                        warnings.warn("Could not find interaction {} with min active >= {}".format(j, min_active))
                    break
        support_idx = np.stack(support_idx).ravel()
        support = np.zeros(p)
        for j in support_idx:
            support[j] = 1
        return y, support

    beta = generate_coef(beta, m)
    if tau == 'median':
        tau = np.median(X,axis = 0)
    
    if min_active is None:
        y_train = np.array([lss_func(X[i, :], beta) for i in range(n)])
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
    else:
        y_train, support = lss_vector_fun(X, beta)

    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn

    if frac_corrupt is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage":
            y_train = corrupt_leverage(X[:, :(m*r)], y_train, frac_corrupt, corrupt_quantile)
            y_train = y_train + sigma * error_fun(n)
  
    if return_support:
        return y_train, support, beta
    else:
        return y_train


def xor(X, sigma, beta, heritability=None, snr=None, error_fun=None):
    n, p = X.shape
    assert p >= 2

    y_train = beta * ((X[:, 0] * X[:, 1]) > 0)
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    y_train = y_train + sigma * error_fun(n)

    return y_train




def partial_linear_lss_model(X, sigma, s, m, r, tau, beta, heritability=None, snr=None, error_fun=None, min_active=None,
              frac_corrupt=None, corrupt_how='permute', corrupt_quantile=None, diagnostics=False, return_support=False):
    """
    This method creates response from an linear + lss model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    s: denotes number of linear terms in EACH interaction term
    tau: threshold
    sigma: standard deviation of noise
    beta: coefficient vector. If beta not a vector, then assumed a constant

    :return
    y_train: numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r  # Cannot have more interactions * size than the dimension
    assert s <= r
    
    def partial_linear_func(x,s,beta):
        y = 0.0
        count = 0
        for j in range(m):
            for i in range(s):
                y += beta[count]*x[j*r+i]
                count += 1
        return y
                

    def lss_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        return y

    def lss_vector_fun(x, beta, beta_linear):
        x_bool = (x - tau) > 0
        y = 0
        max_iter = 100
        features = np.arange(p)
        support_idx = []
        for j in range(m):
            cnt = 0
            while True:
                int_features = np.concatenate(
                    [np.arange(j*r, j*r+s), np.random.choice(features, size=r-s, replace=False)]
                )
                lss_term_components = x_bool[:, int_features]
                lss_term = np.apply_along_axis(all, 1, lss_term_components)
                cnt += 1
                if np.mean(lss_term) >= min_active or cnt > max_iter:
                    norm_constant = sum(np.var(x[:, (j*r):(j*r+s)], axis=0) * beta_linear[(j*s):((j+1)*s)]**2)
                    relative_beta = beta[j] / sum(beta_linear[(j*s):((j+1)*s)])
                    y += lss_term * relative_beta * np.sqrt(norm_constant) / np.std(lss_term)
                    features = list(set(features).difference(set(int_features)))
                    support_idx.append(int_features)
                    if cnt > max_iter:
                        warnings.warn("Could not find interaction {} with min active >= {}".format(j, min_active))
                    break
        support_idx = np.stack(support_idx).ravel()
        support = np.zeros(p)
        for j in support_idx:
            support[j] = 1
        return y, support

    beta_lss = generate_coef(beta, m)
    beta_linear = generate_coef(beta, s*m)
    if tau == 'median':
        tau = np.median(X,axis = 0)

    y_train_linear = np.array([partial_linear_func(X[i, :],s,beta_linear ) for i in range(n)])
    if min_active is None:
        y_train_lss = np.array([lss_func(X[i, :], beta_lss) for i in range(n)])
        support = np.concatenate((np.ones(max(m * r, s)), np.zeros(X.shape[1] - max((m * r), s))))
    else:
        y_train_lss, support = lss_vector_fun(X, beta_lss, beta_linear)
    y_train = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    
    if frac_corrupt is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage":
            y_train = corrupt_leverage(X[:, :max(m*r, s)], y_train, frac_corrupt, corrupt_quantile)
            y_train = y_train + sigma * error_fun(n)
    if return_support:
        return y_train, support, beta_lss
    elif diagnostics:
        return y_train, y_train_linear, y_train_lss
    else:
        return y_train


def linear_lss_model(X, sigma, m, r, tau, beta, s=None, heritability=None, snr=None, error_fun=None,
                     frac_corrupt=None, corrupt_how='permute', corrupt_quantile=None,
                     return_support=False, diagnostics=False):
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
    n, p = X.shape
    if s is None:
        s = m * r

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
    # Make beta vector for LSS
    # beta_lss = np.zeros(m)
    # for j in range(m):
    #     X_block = X[:, j * r: j * r + r]
    #     beta_lin_block = beta_linear[j * r: j * r + r]
    #     X_block_bool = X_block > tau
    #     block_lss_prob = np.all(X_block_bool, axis=1).mean()
    #     block_lss_var = block_lss_prob * (1 - block_lss_prob)
    #     block_lin_var = np.var(X_block @ beta_lin_block)
    #     ratio = np.sqrt(block_lin_var / block_lss_var)
    #     beta_lss[j] = beta * ratio

    y_train_linear = np.array([linear_func(X[i, :], s, beta_linear) for i in range(n)])
    y_train_lss = np.array([lss_func(X[i, :], beta_lss) for i in range(n)])
    y_train = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    
    if frac_corrupt is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage":
            y_train = corrupt_leverage(X[:, :max(m*r, s)], y_train, frac_corrupt, corrupt_quantile)
            y_train = y_train + sigma * error_fun(n)
    #y_train = y_train + sigma * error_fun(n)
    if return_support:
        support = np.concatenate((np.ones(max(m * r, s)), np.zeros(X.shape[1] - max((m * r), s))))
        return y_train, support, beta_lss
    elif diagnostics:
        return y_train, y_train_linear, y_train_lss
    else:
        return y_train

#def hierarchical_lss(X,sigma = None,m = 1, s = 0, beta = 1, heritability = None, snr = None,return_support = False):
#    
#    n, p = X.shape
#    assert p >= m * r
#    
#    def partial_linear_func(x,s,beta):
#        y = 0.0
#        for i in range(s):
#            y += beta[i]*x[i]
#        return y
#    def nested_lss_model(x,m,beta):
#        y = 0.0
#        x_bool = (x - tau) > 0
#        for i in range(m):    
#    if return_support:
#        support = np.concatenate((np.ones(max(m * r, s)), np.zeros(X.shape[1] - max((m * r), s))))
#        return y_train, support, beta_lss
#    elif diagnostics:
#        return y_train, y_train_linear, y_train_lss
#    else:
#        return y_train
                    
def hierarchical_poly(X, sigma=None, m=1, r=1, beta=1, heritability=None, snr=None,
                      frac_corrupt=None, corrupt_how='permute', corrupt_quantile=None,
                      error_fun=None, return_support=False):
    """
    This method creates response from an Linear + LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    s: sparsity 
    sigma: standard deviation of noise
    beta: coefficient vector. If beta not a vector, then assumed a constant

    :return
    y_train: numpy array of shape (n)
    """

    n, p = X.shape
    assert p >= m * r

    def reg_func(x, beta):
        y = 0
        for i in range(m):
            hier_term = 1.0
            for j in range(r):
                hier_term += x[i * r + j] * hier_term
            y += hier_term * beta[i]
        return y

    beta = generate_coef(beta, m)
    y_train = np.array([reg_func(X[i, :], beta) for i in range(n)])
    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn

    if frac_corrupt is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        num_corrupt = int(np.floor(frac_corrupt*len(y_train)))
        corrupt_indices = random.sample([*range(len(y_train))], k=num_corrupt)
        if corrupt_how == 'permute':
            corrupt_array = y_train[corrupt_indices]
            corrupt_array = random.sample(list(corrupt_array), len(corrupt_array))
            for i,index in enumerate(corrupt_indices):
                y_train[index] = corrupt_array[i]
            y_train = y_train + sigma * error_fun(n)           
        elif corrupt_how == 'cauchy':
            for i in range(len(y_train)):
                if i in corrupt_indices:
                    y_train[i] = y_train[i] + sigma*np.random.standard_cauchy()
                else:
                     y_train[i] = y_train[i] + sigma*error_fun()
        elif corrupt_how == "leverage":
            y_train = corrupt_leverage(X[:, :(m*r)], y_train, frac_corrupt, corrupt_quantile)
            y_train = y_train + sigma * error_fun(n)
    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train


    

def model_based_X(X_fun, X_params_dict, y, model, n=None):
    X = X_fun(**X_params_dict)

    if n is not None:
        keep_idx = np.random.choice(X.shape[0], n, replace=False)
        X = IndexedArray(X[keep_idx, :], index=keep_idx)
        y = y[keep_idx]

    model.fit(X, y)
    X = X[:, [i[0] for i in sorted(enumerate(-model.feature_importances_), key=lambda x: x[1])]]
    return X


def model_based_y(X, y, model, sigma, s, heritability=None, snr=None, error_fun=None, return_support=False):
    """
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    y: response vector
    s: sparsity
    sigma: s.d. of added noise
    classification: boolean; whether or not this is a classification problem
    Returns:
    numpy array of shape (n)
    """

    if isinstance(X, IndexedArray):
        y = y[X.index]
    model.fit(X[:, :s], y)
    y_train = model.predict(X[:, :s])

    if heritability is not None:
        sigma = (np.var(y_train) * ((1.0 - heritability) / heritability)) ** 0.5
    if snr is not None:
        sigma = (np.var(y_train) / snr) ** 0.5
    if error_fun is None:
        error_fun = np.random.randn
    y_train = y_train + sigma * error_fun((len(X)))
    if return_support:
        beta = None
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def sample_real_y(X, y, s=None, return_support=False):
    """
    This method is used to sample from a real y
    Parameters:
    X: X matrix
    y: response vector
    s: sparsity
    Returns:
    numpy array of shape (n)
    """

    if isinstance(X, IndexedArray):
        y = y[X.index]

    if return_support:
        beta = None
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y, support, beta
    else:
        return y


class IndexedArray(np.ndarray):
    def __new__(cls, input_array, index=None):
        obj = np.asarray(input_array).view(cls)
        obj.index = index
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.index = getattr(obj, 'index', None)
