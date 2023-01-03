import numpy as np
import pandas as pd
import random
from scipy.linalg import toeplitz
import warnings
import math


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
    if sample_col_n is not None:
        if signal_features is None:
            X = X.sample(n=sample_col_n, replace=False, axis=1)
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


def corrupt_leverage(x_train, y_train, mean_shift, corrupt_quantile, mode="normal"):
    assert mode in ["normal", "constant"]
    if mean_shift is None:
        return y_train
    ranked_rows = np.apply_along_axis(np.linalg.norm, axis=1, arr=x_train).argsort().argsort()
    low_idx = np.where(ranked_rows < round(corrupt_quantile * len(y_train)))[0]
    hi_idx = np.where(ranked_rows >= (len(y_train) - round(corrupt_quantile * len(y_train))))[0]
    if mode == "normal":
        hi_corrupted = np.random.normal(mean_shift, 1, size=len(hi_idx))
        low_corrupted = np.random.normal(-mean_shift, 1, size=len(low_idx))
    elif mode == "constant":
        hi_corrupted = mean_shift
        low_corrupted = -mean_shift
    y_train[hi_idx] = hi_corrupted
    y_train[low_idx] = low_corrupted
    return y_train


def linear_model(X, sigma, s, beta, heritability=None, snr=None, error_fun=None,
                 frac_corrupt=None, corrupt_how='permute', corrupt_size=None, 
                 corrupt_mean=None, return_support=False):
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
    if frac_corrupt is None and corrupt_size is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        if frac_corrupt is None:
            frac_corrupt = 0
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
        elif corrupt_how == "leverage_constant":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(s, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="constant")
        elif corrupt_how == "leverage_normal":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(s, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="normal")

    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train


def lss_model(X, sigma, m, r, tau, beta, heritability=None, snr=None, error_fun=None, min_active=None,
              frac_corrupt=None, corrupt_how='permute', corrupt_size=None, corrupt_mean=None,
              return_support=False):
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

    if frac_corrupt is None and corrupt_size is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        if frac_corrupt is None:
            frac_corrupt = 0
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
        elif corrupt_how == "leverage_constant":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(m*r, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="constant")
        elif corrupt_how == "leverage_normal":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(m*r, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="normal")
  
    if return_support:
        return y_train, support, beta
    else:
        return y_train


def partial_linear_lss_model(X, sigma, s, m, r, tau, beta, heritability=None, snr=None, error_fun=None,
                             min_active=None, frac_corrupt=None, corrupt_how='permute', corrupt_size=None,
                             corrupt_mean=None, diagnostics=False, return_support=False):
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
    
    if frac_corrupt is None and corrupt_size is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        if frac_corrupt is None:
            frac_corrupt = 0
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
        elif corrupt_how == "leverage_constant":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(max(m*r, s), p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="constant")
        elif corrupt_how == "leverage_normal":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(max(m*r, s), p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="normal")
        
    if return_support:
        return y_train, support, beta_lss
    elif diagnostics:
        return y_train, y_train_linear, y_train_lss
    else:
        return y_train

                    
def hierarchical_poly(X, sigma=None, m=1, r=1, beta=1, heritability=None, snr=None,
                      frac_corrupt=None, corrupt_how='permute', corrupt_size=None,
                      corrupt_mean=None, error_fun=None, return_support=False):
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

    if frac_corrupt is None and corrupt_size is None:
        y_train = y_train + sigma * error_fun(n)
    else:
        if frac_corrupt is None:
            frac_corrupt = 0
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
        elif corrupt_how == "leverage_constant":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(m*r, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="constant")
        elif corrupt_how == "leverage_normal":
            if isinstance(corrupt_size, int):
                corrupt_quantile = corrupt_size / n
            else:
                corrupt_quantile = corrupt_size
            y_train = y_train + sigma * error_fun(n)
            corrupt_idx = np.random.choice(range(m*r, p), size=1)
            y_train = corrupt_leverage(X[:, corrupt_idx], y_train, mean_shift=corrupt_mean, corrupt_quantile=corrupt_quantile, mode="normal")
        
    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train


def logistic_model(X, s, beta=None, beta_grid=np.logspace(-4, 4, 100), heritability=None,
                   frac_label_corruption=None, return_support=False):
    """
    This method is used to create responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    Returns:
    numpy array of shape (n)
    """
    
    def create_prob(x, beta):
        linear_term = 0
        for j in range(len(beta)):
            linear_term += x[j] * beta[j]
        prob = 1 / (1 + np.exp(-linear_term))
        return prob

    def create_y(x, beta):
        linear_term = 0
        for j in range(len(beta)):
            linear_term += x[j] * beta[j]
        prob = 1 / (1 + np.exp(-linear_term))
        return (np.random.uniform(size=1) < prob) * 1

    if heritability is None:
        beta = generate_coef(beta, s)
        y_train = np.array([create_y(X[i, :], beta) for i in range(len(X))]).ravel()
    else:
        # find beta to get desired heritability via adaptive grid search within eps=0.01
        y_train, beta, heritability, hdict = logistic_heritability_search(X, heritability, s, create_prob, beta_grid)
        
    if frac_label_corruption is None:
        y_train = y_train
    else:
        corrupt_indices = np.random.choice(np.arange(len(y_train)), size=math.ceil(frac_label_corruption*len(y_train)))
        y_train[corrupt_indices] = 1 - y_train[corrupt_indices]
    if return_support:
        support = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
        return y_train, support, beta
    else:
        return y_train
    

def logistic_lss_model(X, m, r, tau, beta=None, heritability=None, beta_grid=np.logspace(-4, 4, 100),
                       min_active=None, frac_label_corruption=None, return_support=False):
    """
    This method is used to create responses from a logistic model model with lss
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
    Returns:
    numpy array of shape (n)
    """
    n, p = X.shape

    def lss_prob_func(x, beta):
        x_bool = (x - tau) > 0
        y = 0
        for j in range(m):
            lss_term_components = x_bool[j * r:j * r + r]
            lss_term = int(all(lss_term_components))
            y += lss_term * beta[j]
        prob = 1 / (1 + np.exp(-y))
        return prob
    
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

    if tau == 'median':
        tau = np.median(X,axis = 0)
    
    if heritability is None:
        beta = generate_coef(beta, m)
        if min_active is None:
            y_train = np.array([lss_func(X[i, :], beta) for i in range(n)]).ravel()
            support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        else:
            y_train, support = lss_vector_fun(X, beta)
            y_train = y_train.ravel()
    else:
        if min_active is not None:
            raise ValueError("Cannot set heritability and min_active at the same time.")
        # find beta to get desired heritability via adaptive grid search within eps=0.01 (need to jitter beta to reach higher signals)
        y_train, beta, heritability, hdict = logistic_heritability_search(X, heritability, m, lss_prob_func, beta_grid, jitter_beta=True)
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
    
    if frac_label_corruption is None:
        y_train = y_train
    else:
        corrupt_indices = np.random.choice(np.arange(len(y_train)), size=math.ceil(frac_label_corruption*len(y_train)))
        y_train[corrupt_indices] = 1 - y_train[corrupt_indices]

    if return_support:
        return y_train, support, beta
    else:
        return y_train


def logistic_partial_linear_lss_model(X, s, m, r, tau, beta=None, heritability=None, beta_grid=np.logspace(-4, 4, 100),
                                      min_active=None, frac_label_corruption=None, return_support=False):
    """
    This method is used to create responses from a logistic model model with lss
    X: X matrix
    s: sparsity
    beta: coefficient vector. If beta not a vector, then assumed a constant
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
        
    def logistic_prob_func(y):
        prob = 1 / (1 + np.exp(-y))
        return prob

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

    if tau == 'median':
        tau = np.median(X,axis = 0)

    if heritability is None:
        beta_lss = generate_coef(beta, m)
        beta_linear = generate_coef(beta, s*m)
        
        y_train_linear = np.array([partial_linear_func(X[i, :],s,beta_linear ) for i in range(n)])
        if min_active is None:
            y_train_lss = np.array([lss_func(X[i, :], beta_lss) for i in range(n)])
            support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        else:
            y_train_lss, support = lss_vector_fun(X, beta_lss, beta_linear)
        y_train = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])
        y_train = np.array([logistic_link_func(y_train[i]) for i in range(n)])
    else:
        if min_active is not None:
            raise ValueError("Cannot set heritability and min_active at the same time.")
        # find beta to get desired heritability via adaptive grid search within eps=0.01
        eps = 0.01
        max_iter = 1000
        pves = {}
        for idx, beta in enumerate(beta_grid):
            beta_lss_vec = generate_coef(beta, m)
            beta_linear_vec = generate_coef(beta, s*m)

            y_train_linear = np.array([partial_linear_func(X[i, :], s, beta_linear_vec) for i in range(n)])
            y_train_lss = np.array([lss_func(X[i, :], beta_lss_vec) for i in range(n)])
            y_train_sum = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])
            prob_train = np.array([logistic_prob_func(y_train_sum[i]) for i in range(n)]).ravel()
            np.random.seed(idx)
            y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
            pve = np.var(prob_train) / np.var(y_train)
            pves[(idx, beta)] = pve

        (idx, beta), pve = min(pves.items(), key=lambda x: abs(x[1] - heritability))
        beta_lss_vec = generate_coef(beta, m)
        beta_linear_vec = generate_coef(beta, s*m)

        y_train_linear = np.array([partial_linear_func(X[i, :], s, beta_linear_vec) for i in range(n)])
        y_train_lss = np.array([lss_func(X[i, :], beta_lss_vec) for i in range(n)])
        y_train_sum = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])

        prob_train = np.array([logistic_prob_func(y_train_sum[i]) for i in range(n)]).ravel()
        np.random.seed(idx)
        y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
        if pve > heritability:
            min_beta = beta_grid[idx-1]
            max_beta = beta
        else:
            min_beta = beta
            max_beta = beta_grid[idx+1]
        cur_beta = (min_beta + max_beta) / 2
        iter = 1
        while np.abs(pve - heritability) > eps:
            beta_lss_vec = generate_coef(cur_beta, m)
            beta_linear_vec = generate_coef(cur_beta, s*m)

            y_train_linear = np.array([partial_linear_func(X[i, :], s, beta_linear_vec) for i in range(n)])
            y_train_lss = np.array([lss_func(X[i, :], beta_lss_vec) for i in range(n)])
            y_train_sum = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])

            prob_train = np.array([logistic_prob_func(y_train_sum[i]) for i in range(n)]).ravel()
            np.random.seed(iter + len(beta_grid))
            y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
            pve = np.var(prob_train) / np.var(y_train)
            pves[(iter + len(beta_grid), cur_beta)] = pve
            if pve > heritability:
                max_beta = cur_beta
            else:
                min_beta = cur_beta
            beta = cur_beta
            cur_beta = (min_beta + max_beta) / 2
            iter += 1
            if iter > max_iter:
                (idx, cur_beta), pve = min(pves.items(), key=lambda x: abs(x[1] - heritability))
                beta_lss_vec = generate_coef(cur_beta, m)
                beta_linear_vec = generate_coef(cur_beta, s*m)

                y_train_linear = np.array([partial_linear_func(X[i, :], s, beta_linear_vec) for i in range(n)])
                y_train_lss = np.array([lss_func(X[i, :], beta_lss_vec) for i in range(n)])
                y_train_sum = np.array([y_train_linear[i] + y_train_lss[i] for i in range(n)])

                prob_train = np.array([logistic_prob_func(y_train_sum[i]) for i in range(n)]).ravel()
                np.random.seed(idx)
                y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
                pve = np.var(prob_train) / np.var(y_train)
                beta = cur_beta
                break
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))

    if frac_label_corruption is None:
        y_train = y_train
    else:
        corrupt_indices = np.random.choice(np.arange(len(y_train)), size=math.ceil(frac_label_corruption*len(y_train)))
        y_train[corrupt_indices] = 1 - y_train[corrupt_indices]

    y_train = y_train.ravel()
    
    if return_support:
        return y_train, support, beta
    else:
        return y_train

    
def logistic_hier_model(X, m, r, beta=None, heritability=None, beta_grid=np.logspace(-4, 4, 100),
                        frac_label_corruption=None, return_support=False):
    
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
        
    def prob_func(x, beta):
        y = 0
        for i in range(m):
            hier_term = 1.0
            for j in range(r):
                hier_term += x[i * r + j] * hier_term
            y += hier_term * beta[i]
        return 1 / (1 + np.exp(-y))
    
    if heritability is None:
        beta = generate_coef(beta, m)
        y_train = np.array([reg_func(X[i, :], beta) for i in range(n)])
        y_train = np.array([logistic_link_func(y_train[i]) for i in range(n)])
    else:
        # find beta to get desired heritability via adaptive grid search within eps=0.01
        y_train, beta, heritability, hdict = logistic_heritability_search(X, heritability, m, prob_func, beta_grid)
    
    if frac_label_corruption is None:
        y_train = y_train
    else:
        corrupt_indices = np.random.choice(np.arange(len(y_train)), size=math.ceil(frac_label_corruption*len(y_train)))
        y_train[corrupt_indices] = 1 - y_train[corrupt_indices]
    y_train = y_train.ravel()
    
    if return_support:
        support = np.concatenate((np.ones(m * r), np.zeros(X.shape[1] - (m * r))))
        return y_train, support, beta
    else:
        return y_train
      
      
def logistic_heritability_search(X, heritability, s, prob_fun, beta_grid=np.logspace(-4, 4, 100),
                                 eps=0.01, max_iter=1000, jitter_beta=False, return_pve=True):
    pves = {}

    # first search over beta grid
    for idx, beta in enumerate(beta_grid):
        np.random.seed(idx)
        beta_vec = generate_coef(beta, s)
        if jitter_beta:
            beta_vec = beta_vec + np.random.uniform(-1e-4, 1e-4, beta_vec.shape)
        prob_train = np.array([prob_fun(X[i, :], beta_vec) for i in range(len(X))]).ravel()
        y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
        pve = np.var(prob_train) / np.var(y_train)
        pves[(idx, beta)] = pve

    # find beta with heritability closest to desired heritability
    (idx, beta), pve = min(pves.items(), key=lambda x: abs(x[1] - heritability))
    np.random.seed(idx)
    beta_vec = generate_coef(beta, s)
    if jitter_beta:
        beta_vec = beta_vec + np.random.uniform(-1e-4, 1e-4, beta_vec.shape)
    prob_train = np.array([prob_fun(X[i, :], beta_vec) for i in range(len(X))]).ravel()
    y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1

    # search nearby beta to get closer to desired heritability
    if pve > heritability:
        min_beta = beta_grid[idx-1]
        max_beta = beta
    else:
        min_beta = beta
        max_beta = beta_grid[idx+1]
    cur_beta = (min_beta + max_beta) / 2
    iter = 1
    while np.abs(pve - heritability) > eps:
        np.random.seed(iter + len(beta_grid))
        beta_vec = generate_coef(cur_beta, s)
        if jitter_beta:
            beta_vec = beta_vec + np.random.uniform(-1e-4, 1e-4, beta_vec.shape)
        prob_train = np.array([prob_fun(X[i, :], beta_vec) for i in range(len(X))]).ravel()
        y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
        pve = np.var(prob_train) / np.var(y_train)
        pves[(iter + len(beta_grid), cur_beta)] = pve
        if pve > heritability:
            max_beta = cur_beta
        else:
            min_beta = cur_beta
        cur_beta = (min_beta + max_beta) / 2
        beta = beta_vec
        iter += 1
        if iter > max_iter:
            (idx, cur_beta), pve = min(pves.items(), key=lambda x: abs(x[1] - heritability))
            np.random.seed(idx)
            beta_vec = generate_coef(cur_beta, s)
            if jitter_beta:
                beta_vec = beta_vec + np.random.uniform(-1e-4, 1e-4, beta_vec.shape)
            prob_train = np.array([prob_fun(X[i, :], beta_vec) for i in range(len(X))]).ravel()
            y_train = (np.random.uniform(size=len(prob_train)) < prob_train) * 1
            pve = np.var(prob_train) / np.var(y_train)
            beta = beta_vec
            break

    if return_pve:
        return y_train, beta, pve, pves
    else:
        return y_train, beta


def entropy_X(n, scale=False):
    x1 = np.random.choice([0, 1], (n, 1), replace=True)
    x2 = np.random.normal(0, 1, (n, 1))
    x3 = np.random.choice(np.arange(4), (n, 1), replace=True)
    x4 = np.random.choice(np.arange(10), (n, 1), replace=True)
    x5 = np.random.choice(np.arange(20), (n, 1), replace=True)
    X = np.concatenate((x1, x2, x3, x4, x5), axis=1)
    if scale:
        X = (X - X.mean()) / X.std()
    return X


def entropy_y(X, c=3, return_support=False):
    if any(X[:, 0] < 0):
        x = (X[:, 0] > 0) * 1
    else:
        x = X[:, 0]
    prob = ((c - 2) * x + 1) / c
    y = (np.random.uniform(size=len(prob)) < prob) * 1
    if return_support:
        support = np.array([0, 1, 0, 0, 0])
        beta = None
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

#%%
