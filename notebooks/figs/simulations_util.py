from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def sample_boolean_X(n, d):
    X = np.random.randint(0, 2.0, (n, d))
    return X


def linear_model(X, sigma, s, beta):
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


def sum_of_squares(X, sigma, s, beta):
    '''
    This method is used to create responses from a sum of squares model with hard sparsity
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


def lss_model(X, sigma, m, r, tau):
    """
    This method creates response from an LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    tau: threshold
    sigma: standard deviation of noise

    :return
    y_train: numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r  # Cannot have more interactions * size than the dimension

    def lss_func(x):
        x_bool = (x - tau) > 0
        y = 0
        for i in range(m):
            lss_term_components = x_bool[i * r:i * r + r]
            lss_term = all(lss_term_components)
            y += lss_term
        return y

    y_train = np.array([lss_func(X[i, :]) for i in range(n)])
    y_train = y_train + sigma * np.random.randn(n)

    return y_train


def sum_of_polys(X, sigma, m, r):
    """
    This method creates response from an LSS model

    X: data matrix
    m: number of interaction terms
    r: max order of interaction
    sigma: standard deviation of noise

    :return
    y_train: numpy array of shape (n)
    """
    n, p = X.shape
    assert p >= m * r  # Cannot have more interactions * size than the dimension

    def poly_func(x):
        y = 0
        for i in range(m):
            poly_term_components = x[i * r:i * r + r]
            poly_term = np.prod(poly_term_components)
            y += poly_term
        return y

    y_train = np.array([poly_func(X[i, :]) for i in range(n)])
    y_train = y_train + sigma * np.random.randn(n)

    return y_train


def get_best_fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    return [m, b]


def is_leaf(node):
    return node.left is None and node.right is None


def print_tree(node, depth):
    """
    Prints part of a tree, starting with a node at depth depth.

    :param node:
    :param depth:
    :return:
    """

    branch = "o--"
    prev_branch = "|  "

    def make_node_str(node, depth):
        return_str = ""
        return_str += prev_branch * (depth - 1)
        return_str += branch * (depth > 0)
        return_str += str(node)
        return return_str

    if is_leaf(node):
        print(make_node_str(node, depth))
    else:
        print(make_node_str(node, depth))
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)


def get_split_feats(node):
    if node is None:
        return set({})
    elif is_leaf(node):
        return set({})
    else:
        return set({node.feature}).union(get_split_feats(node.left)) \
            .union(get_split_feats(node.right))


def set_to_matrix(feats_set, d):
    indicator_vec = np.zeros(d)
    for feat in feats_set:
        if feat < d:
            indicator_vec[feat] = 1

    return np.outer(indicator_vec, indicator_vec)


def get_feat_cooccurence(trees, d):
    num_trees = len(trees)
    cooccurence_matrix = np.zeros((d, d))
    for tree in trees:
        split_feats = get_split_feats(tree)
        split_mat = set_to_matrix(split_feats, d)
        cooccurence_matrix += split_mat

    return cooccurence_matrix / num_trees


def get_split_feats_counts(node):
    if node is None:
        return []
    elif is_leaf(node):
        return []
    else:
        return [node.feature] + get_split_feats_counts(node.left) + get_split_feats_counts(node.right)


def get_feat_counts_matrix(trees, d):
    num_trees = len(trees)
    count_matrix = np.zeros((num_trees, d))
    for i, tree in enumerate(trees):
        split_counts = Counter(get_split_feats_counts(tree))
        for k, v in split_counts.items():
            if k < d:
                count_matrix[i, k] = v

    return count_matrix


def get_feat_counts_cossim(trees, d):
    count_matrix = get_feat_counts_matrix(trees, d)
    counts_cossim = np.zeros((d, d))
    for i in range(d):
        counts_cossim[i, i] = 1
        for j in range(i):
            counts_cossim[i, j] = 1 - cosine(count_matrix[:, i], count_matrix[:, j])
            counts_cossim[j, i] = counts_cossim[i, j]

    return counts_cossim


def get_feat_counts_correlation(trees, d):
    count_matrix = get_feat_counts_matrix(trees, d)
    counts_correlation = pd.DataFrame(count_matrix).corr()

    return counts_correlation
