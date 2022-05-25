import numpy as np
import warnings
import sklearn

from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import scale


def MDI_OOB(rf, X, y, type='oob', normalized=True, balanced=False, demean=False, normal_fX=False):
    n_samples, n_features = X.shape
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')
    out = np.zeros((n_features,))
    SE = np.zeros((n_features,))
    if demean:
        # demean y
        y = y - np.mean(y, axis=0)

    for tree in rf.estimators_:
        if type == 'oob':
            if rf.bootstrap:
                indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples)
            else:
                raise ValueError('Without bootstrap, it is not possible to calculate oob.')
        elif type == 'test':
            indices = np.arange(n_samples)
        elif type == 'classic':
            if rf.bootstrap:
                indices = _generate_sample_indices(tree.random_state, n_samples, n_samples)
            else:
                indices = np.arange(n_samples)
        else:
            raise ValueError('type is not recognized. (%s)' % (type))
        _, _, contributions = _predict_tree(tree, X[indices, :])
        if balanced and (type == 'oob' or type == 'test'):
            base_indices = _generate_sample_indices(tree.random_state, n_samples, n_samples)
            ids = tree.apply(X[indices, :])
            base_ids = tree.apply(X[base_indices, :])
            tmp1, tmp2 = np.unique(ids, return_counts=True)
            weight1 = {key: 1. / value for key, value in zip(tmp1, tmp2)}
            tmp1, tmp2 = np.unique(base_ids, return_counts=True)
            weight2 = {key: value for key, value in zip(tmp1, tmp2)}
            final_weights = np.array([[weight1[id] * weight2[id]] for id in ids])
            final_weights /= np.mean(final_weights)
        else:
            final_weights = 1
        if len(contributions.shape) == 2:
            contributions = contributions[:, :, np.newaxis]
        # print(contributions.shape, y[indices,:].shape)
        if normal_fX:
            for k in range(contributions.shape[-1]):
                contributions[:, :, k] = scale(contributions[:, :, k])
        tmp = np.tensordot(np.array(y[indices, :]) * final_weights, contributions, axes=([0, 1], [0, 2]))
        if normalized:
            if sum(tmp) != 0:
                out += tmp / sum(tmp)
        else:
            out += tmp / len(indices)
        if normalized:
            if sum(tmp) != 0:
                SE += (tmp / sum(tmp)) ** 2
        else:
            SE += (tmp / len(indices)) ** 2
    out /= rf.n_estimators
    SE /= rf.n_estimators
    SE = ((SE - out ** 2) / rf.n_estimators) ** .5
    return out, SE


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


def _predict_tree(model, X, joint_contribution=False):
    """
    For a given DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, or ExtraTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    leaves = model.apply(X)
    paths = _get_tree_paths(model.tree_, 0)

    for path in paths:
        path.reverse()

    leaf_to_path = {}
    # map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path

        # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze(axis=1)
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
    if isinstance(model, DecisionTreeRegressor):
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
    elif isinstance(model, DecisionTreeClassifier):
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], model.n_classes_)
    else:
        warnings.warn('the instance is not recognized. Try to proceed with classifier but could fail.')
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], model.n_classes_)

    direct_prediction = values[leaves]

    # make into python list, accessing values will be faster
    values_list = list(values)
    feature_index = list(model.tree_.feature)

    contributions = []
    if joint_contribution:
        for row, leaf in enumerate(leaves):
            path = leaf_to_path[leaf]

            path_features = set()
            contributions.append({})
            for i in range(len(path) - 1):
                path_features.add(feature_index[path[i]])
                contrib = values_list[path[i + 1]] - \
                          values_list[path[i]]
                # path_features.sort()
                contributions[row][tuple(sorted(path_features))] = \
                    contributions[row].get(tuple(sorted(path_features)), 0) + contrib
        return direct_prediction, biases, contributions

    else:
        unique_leaves = np.unique(leaves)
        unique_contributions = {}

        for row, leaf in enumerate(unique_leaves):
            for path in paths:
                if leaf == path[-1]:
                    break

            contribs = np.zeros(line_shape)
            for i in range(len(path) - 1):
                contrib = values_list[path[i + 1]] - \
                          values_list[path[i]]
                contribs[feature_index[path[i]]] += contrib
            unique_contributions[leaf] = contribs

        for row, leaf in enumerate(leaves):
            contributions.append(unique_contributions[leaf])

        return direct_prediction, biases, np.array(contributions)


def _predict_forest(model, X, joint_contribution=False):
    """
    For a given RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
    [prediction, bias and feature_contributions], such that prediction ≈ bias +
    feature_contributions.
    """
    biases = []
    contributions = []
    predictions = []

    if joint_contribution:

        for tree in model.estimators_:
            pred, bias, contribution = _predict_tree(tree, X, joint_contribution=joint_contribution)

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)

        total_contributions = []

        for i in range(len(X)):
            contr = {}
            for j, dct in enumerate(contributions):
                for k in set(dct[i]).union(set(contr.keys())):
                    contr[k] = (contr.get(k, 0) * j + dct[i].get(k, 0)) / (j + 1)

            total_contributions.append(contr)

        for i, item in enumerate(contribution):
            total_contributions[i]
            sm = sum([v for v in contribution[i].values()])

        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
                total_contributions)
    else:
        for tree in model.estimators_:
            pred, bias, contribution = _predict_tree(tree, X)

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)

        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
                np.mean(contributions, axis=0))


def predict(model, X, joint_contribution=False):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier,
        ExtraTreeRegressor, ExtraTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier
    Scikit-learn model on which the prediction should be decomposed.
    X : array-like, shape = (n_samples, n_features)
    Test samples.
    
    joint_contribution : boolean
    Specifies if contributions are given individually from each feature,
    or jointly over them
    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, If joint_contribution is False then returns and  array of 
        shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification, denoting
        contribution from each feature.
        If joint_contribution is True, then shape is array of size n_samples,
        where each array element is a dict from a tuple of feature indices to
        to a value denoting the contribution from that feature tuple.
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if (isinstance(model, DecisionTreeClassifier) or
            isinstance(model, DecisionTreeRegressor)):
        return _predict_tree(model, X, joint_contribution=joint_contribution)
    elif (isinstance(model, ForestClassifier) or
          isinstance(model, ForestRegressor)):
        return _predict_forest(model, X, joint_contribution=joint_contribution)
    else:
        raise ValueError("Wrong model type. Base learner needs to be a "
                         "DecisionTreeClassifier or DecisionTreeRegressor.")
