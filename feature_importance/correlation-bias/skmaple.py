from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np


class MAPLE(BaseEstimator):
    def __init__(self, estimator, linear_model):
        self.estimator = estimator
        self.linear_model = linear_model

    def fit(self, X, y):
        X = check_array(X)
        y = column_or_1d(y, warn=True)

        if not hasattr(self.estimator, 'apply'):
            raise ValueError('{} should have apply function'.format(
                self.estimator))
            
        self.estimator.fit(X, y)
        self.X_fit_ = X
        self.y_fit_ = y
        self.train_leaf_ix_ = self.estimator.apply(X)

    def _get_weights(self, xi):
        K = self.estimator.n_estimators
        leaf_ix = self.estimator.apply([xi])[0]
        weights = []
        for ix in self.train_leaf_ix_:
            weights.append(sum(ix == leaf_ix) / K)
        return weights

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['X_fit_', 'y_fit_'])

        self.fitted_linear_models_ = []
        self.weights_ = []
        preds = []

        for xi in X:
            weights = self._get_weights(xi)

            model = clone(self.linear_model)
            model.fit(self.X_fit_, self.y_fit_, sample_weight=weights)

            self.weights_.append(weights)
            self.fitted_linear_models_.append(model)
            preds.append(model.predict([xi])[0])

        return preds
