import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import pandas as pd

def tp(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[1][1]


def fp(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[0][1]


def neg(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[0][0] + conf_mat[0][1]


def pos(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[1][0] + conf_mat[1][1]


def specificity_score(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])


def pr_auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def MDI(X, y, fit):
    return pd.DataFrame(data={"var": list(range(X.shape[1])),
                              "importance": np.random.randn(X.shape[1])})
