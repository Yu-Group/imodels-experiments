from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder


def MDA(rf, X, y, type = 'oob', n_trials = 10, metric = 'accuracy'):
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')

    y_mda = copy.deepcopy(y)
    if rf._estimator_type == "classifier" and y.dtype == "object":
        y_mda = LabelEncoder().fit(y_mda.ravel()).transform(y_mda.ravel()).reshape(y_mda.shape[0], 1)

    n_samples, n_features = X.shape
    fi_mean = np.zeros((n_features,))
    fi_std = np.zeros((n_features,))
    best_score = rf_accuracy(rf, X, y_mda, type = type, metric = metric)
    for f in range(n_features):
        permute_score = 0
        permute_std = 0
        X_permute = X.copy()
        for i in range(n_trials):
            X_permute[:, f] = np.random.permutation(X_permute[:, f])
            to_add = rf_accuracy(rf, X_permute, y_mda, type = type, metric = metric)
            permute_score += to_add
            permute_std += to_add ** 2
        permute_score /= n_trials
        permute_std /= n_trials
        permute_std = (permute_std - permute_score ** 2) ** .5 / n_trials ** .5
        fi_mean[f] = best_score - permute_score
        fi_std[f] = permute_std
    return fi_mean, fi_std


def neg_mse(y, y_hat):
    return - mean_squared_error(y, y_hat)


def rf_accuracy(rf, X, y, type = 'oob', metric = 'accuracy'):
    if metric == 'accuracy':
        score = accuracy_score
    elif metric == 'mse':
        score = neg_mse
    else:
        raise ValueError('metric type not understood')

    n_samples, n_features = X.shape
    tmp = 0
    count = 0
    if type == 'test':
        return score(y, rf.predict(X))
    elif type == 'train' and not rf.bootstrap:
        return score(y, rf.predict(X))

    for tree in rf.estimators_:
        if type == 'oob':
            if rf.bootstrap:
                indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples)
            else:
                raise ValueError('Without bootstrap, it is not possible to calculate oob.')
        elif type == 'train':
            indices = _generate_sample_indices(tree.random_state, n_samples, n_samples)
        else:
            raise ValueError('type is not recognized. (%s)'%(type))
        tmp += score(y[indices,:], tree.predict(X[indices, :])) * len(indices)
        count += len(indices)
    return tmp / count