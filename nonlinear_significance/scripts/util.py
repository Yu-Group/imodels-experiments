import numpy as np
from collections import defaultdict
from joblib import delayed, Parallel

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
import statistics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def compare(data, feature, threshold, sign=True):
    """
    :param data: A 2D array of size (n_sample, n_feat)
    :param feature:
    :param threshold:
    :param sign:
    :return:
    """
    if sign:
        return data[:, feature] > threshold
    else:
        return data[:, feature] <= threshold


class LocalDecisionStump:

    def __init__(self, feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs):
        """

        :param feature:
        :param threshold:
        :param left_val:
        :param right_val:
        :param a_features: list, comprising ancestor features
        :param a_thresholds: list, comprising ancestor thresholds
        :param a_signs: list, comprising ancestor signs (0: left, 1: right)
        """

        self.feature = feature
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val
        self.a_features = a_features
        self.a_thresholds = a_thresholds
        self.a_signs = a_signs

    def __call__(self, data):
        """

        :param data: A 2D array of query values
        :return:
        """

        root_to_stump_path_indicators = np.zeros((data.shape[0], len(self.a_features))).astype(bool)
        for idx, (f, t, g) in enumerate(zip(self.a_features, self.a_thresholds, self.a_signs)):
            root_to_stump_path_indicators[:, idx] = compare(data, f, t, g)
        in_node = np.all(root_to_stump_path_indicators, axis=1).astype(int)
        # in_node = all([compare(data, f, t, g) for f, t, g in zip(self.a_features,
        #                                                          self.a_thresholds,
        #                                                          self.a_signs)])
        is_right = compare(data, self.feature, self.threshold).astype(int)
        result = in_node * (is_right * self.right_val + (1 - is_right) * self.left_val)
        # if not in_node:
        #     return 0.0
        # else:
        #     is_right = compare(data, self.feature, self.threshold)
        #     if is_right:
        #         return self.right_val
        #     else:
        #         return self.left_val
        return result

    def __repr__(self):
        return f"LocalDecisionStump(feature={self.feature}, threshold={self.threshold}, left_val={self.left_val}, " \
               f"right_val={self.right_val}, a_features={self.a_features}, a_thresholds={self.a_thresholds}, " \
               f"a_signs={self.a_signs})"


def make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize=False):
    """
    Create a single localized decision stump corresponding to a node

    :param node_no:
    :param tree_struct:
    :param parent_stump:
    :param is_right_child:
    :param normalize:
    :return:
    """
    if parent_stump is None:  # If root node
        a_features = []
        a_thresholds = []
        a_signs = []
    else:
        a_features = parent_stump.a_features + [parent_stump.feature]
        a_thresholds = parent_stump.a_thresholds + [parent_stump.threshold]
        a_signs = parent_stump.a_signs + [is_right_child]

    feature = tree_struct.feature[node_no]
    threshold = tree_struct.threshold[node_no]

    if not normalize:
        return LocalDecisionStump(feature, threshold, -1, 1, a_features, a_thresholds, a_signs)
    else:
        # parent_size = tree_struct.n_node_samples[node_no]
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        left_size = tree_struct.n_node_samples[left_child]
        right_size = tree_struct.n_node_samples[right_child]
        left_val = - np.sqrt(right_size / left_size)
        right_val = np.sqrt(left_size / right_size)
        return LocalDecisionStump(feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs)


def make_stumps(tree_struct, normalize=False):
    """
    Take sklearn decision tree structure and create a collection of local
    decision stump objects

    :param tree_struct:
    :param normalize:
    :return: list of stumps
    """
    stumps = []

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child, normalize, stumps):

        new_stump = make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize)
        stumps.append(new_stump)
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        if tree_struct.feature[left_child] != -2:  # is not leaf
            make_stump_iter(left_child, tree_struct, new_stump, False, normalize, stumps)
        if tree_struct.feature[right_child] != -2:  # is not leaf
            make_stump_iter(right_child, tree_struct, new_stump, True, normalize, stumps)

    make_stump_iter(0, tree_struct, None, None, normalize, stumps)

    return stumps


def tree_feature_transform(stumps, X):
    transformed_feature_vectors = []
    for stump in stumps:
        transformed_feature_vec = stump(X)
        transformed_feature_vectors.append(transformed_feature_vec)

    return np.vstack(transformed_feature_vectors).T


class TreeTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, estimator, max_components=np.inf, normalize=True):
        self.estimator = estimator
        self.max_components = max_components
        self.normalize = normalize
        if isinstance(estimator, BaseEnsemble):
            self.all_stumps = []
            for tree_model in estimator.estimators_:
                self.all_stumps += make_stumps(tree_model.tree_, normalize)
        else:
            self.all_stumps = make_stumps(estimator.tree_, normalize)
        self.original_feat_to_stump_mapping = defaultdict(list)
        for idx, stump in enumerate(self.all_stumps):
            self.original_feat_to_stump_mapping[stump.feature].append(idx)
        self.pca_transformers = defaultdict(lambda: None)
        self.original_feat_to_transformed_mapping = defaultdict(list)

    def fit(self, X, y=None, parallelize=False):

        def pca_on_stumps(origin_feat):
            restricted_stumps = [self.all_stumps[idx] for idx in self.original_feat_to_stump_mapping[origin_feat]]
            n_stumps = len(restricted_stumps)
            if n_stumps > self.max_components:
                transformed_feature_vectors = tree_feature_transform(restricted_stumps, X)
                pca = PCA(n_components=self.max_components)
                pca.fit(transformed_feature_vectors)
            else:
                pca = None
            if self.max_components <= 1.0:
                n_new_feats = min(pca.explained_variance_.shape[0], n_stumps)
            else:
                n_new_feats = min(self.max_components, n_stumps)
            return pca, n_new_feats

        n_orig_feats = X.shape[1]
        if parallelize:
            results = Parallel(n_jobs=8)(delayed(pca_on_stumps)(k) for k in np.arange(n_orig_feats))
            counter = 0
            for k in np.arange(n_orig_feats):
                self.pca_transformers[k], n_new_feats_for_k = results[k]
                self.original_feat_to_transformed_mapping[k] = np.arange(counter, counter + n_new_feats_for_k)
                counter += n_new_feats_for_k

        else:
            counter = 0
            for k in np.arange(n_orig_feats):
                self.pca_transformers[k], n_new_feats_for_k = pca_on_stumps(k)
                self.original_feat_to_transformed_mapping[k] = np.arange(counter, counter + n_new_feats_for_k)
                counter += n_new_feats_for_k

    def transform(self, X):
        transformed_feature_vectors_sets = []
        for k, v in self.original_feat_to_stump_mapping.items():
            restricted_stumps = [self.all_stumps[idx] for idx in v]
            transformed_feature_vectors = tree_feature_transform(restricted_stumps, X)
            if self.pca_transformers[k] is not None:
                transformed_feature_vectors = self.pca_transformers[k].transform(transformed_feature_vectors)
            transformed_feature_vectors_sets.append(transformed_feature_vectors)

        return np.hstack(transformed_feature_vectors_sets)

    def get_transformed_X_for_feat(self, X_transformed, feature, max_components):
        '''
        This method takes in the transformed X matrix (applying the node basis) and returns the X matrix restricted to a particular feature. 
        '''
        X_transformed_feat = X_transformed[:, self.original_feat_to_transformed_mapping[feature]]
        if max_components <= X_transformed_feat.shape[1]:
            return X_transformed[:, self.original_feat_to_transformed_mapping[feature]]
        else:
            return X_transformed[:, self.original_feat_to_stump_mapping[feature]]

    # def multiple_test_correction

    # def get_subspace_correlation(self,X_transformed,feat_1,feat_2,max_components = np.inf):
    #    '''
    #    This method takes in the transformed X matrix and returns the subspace correlation matrix. 
    #    '''
    #    feat_1_X =  self.get_transformed_X_for_feat(X_transformed,feat_1,max_components)
    #    feat_2_X =  self.get_transformed_X_for_feat(X_transformed,feat_2,max_components)
    #    feat12_angles = subspace_angles(feat_1_X,feat_2_X)
    #    return np.sum(np.cos(feat12_angles)**2)/len(feat12_angles)
