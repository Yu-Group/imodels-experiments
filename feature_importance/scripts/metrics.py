import numpy as np
from scipy.stats import pearsonr


def corr(y_true, y_pred):
  if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
    return 0
  return pearsonr(y_true, y_pred)[0]


def metric_multiclass_wrapper(metric_fun):

    def multiclass_metric_fun(y_onehot, yhat, sample_weight=None):
        fi_scores = np.zeros(y_onehot.shape[1])
        for i in range(y_onehot.shape[1]):
            fi_scores[i] = metric_fun(y_onehot[:, i], yhat[:, i])
        return fi_scores
    return multiclass_metric_fun
