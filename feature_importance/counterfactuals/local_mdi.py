import numpy as np
import sklearn.ensemble

def compute_mdi_local_tree(tree, X, vimp):
    nsamples, nfeatures = X.shape

    impurity = tree.impurity
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature

    for i in range(nsamples):
        node = 0
        oldvimp = impurity[node]

        while children_left[node] != -1:
            ifeat = features[node]
            if X[i, ifeat] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
            newvimp = impurity[node]
            vimp[i, ifeat] += oldvimp - newvimp
            oldvimp = newvimp


def compute_mdi_local_ens(ens, X, verbose=0):
    nsamples, nfeatures = X.shape
    # vimp = np.zeros((nsamples, ens.n_features_), dtype='float64')
    vimp = np.zeros((nsamples, nfeatures), dtype='float64')

    for i, est in enumerate(ens.estimators_):
        if verbose > 0:
            print("o", end='', flush=True)
        compute_mdi_local_tree(est.tree_, X, vimp)

    if verbose > 0:
        print("")

    vimp /= ens.n_estimators
    return vimp

def local_mdi_score(X_train, X_test, model=None, absolute=True):
    lfi_train = compute_mdi_local_ens(model, X_train)
    lfi_test = compute_mdi_local_ens(model, X_test)
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test