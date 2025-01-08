# import numpy as np
# import pandas as pd
# from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier, RandomForestPlusRegressor
# from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  AloRFPlusMDI, RFPlusMDI
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import SGDRegressor
# from imodels import get_clean_dataset
# from sklearn.linear_model import RidgeCV
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.ensemble import RandomForestClassifier
# import os
# from os.path import join as oj

# # sample train and test data from diabetes dataset
# X, y, feature_names = get_clean_dataset('diabetes')
# # train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # fit random forest with params from MDI+
# rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
#                             random_state=42)

# # ridge rf+
# rf_plus_ridge = RandomForestPlusClassifier(rf_model=rf,
#                     prediction_model=LogisticRegressionCV(penalty='l2',
#                                 cv=5, max_iter=10000, random_state=42))
# rf_plus_ridge.fit(X_train, y_train)

# lmdi_explainer = RFPlusMDI(rf_plus_ridge, mode = "only_k", evaluate_on = "all")

# lmdi_scores = lmdi_explainer.explain_linear_partial(X_train, y_train, ranking=True)

# result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
#                     'results/')

# # write numpy array lmdi_scores to csv
# np.savetxt('jsteinhardt.csv', lmdi_scores, delimiter=',')


# # lmdi_scores.to_csv(oj(result_dir, "jsteinhardt.csv"), index=False)

import numpy as np
import pandas as pd
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  AloRFPlusMDI, RFPlusMDI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from imodels import get_clean_dataset
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import random
from scipy.linalg import toeplitz
import warnings
import math
import imodels
import openml
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def sample_real_data_X(source=None, data_name=None, file_path=None, task_id=None, data_id=None, seed=4307, normalize=False, sample_row_n=None):
    if source == "imodels":
        X, _, _ = imodels.get_clean_dataset(data_name)
    elif source == "openml":
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, _, _, _ = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
    elif source == "uci":
        dataset = fetch_ucirepo(id=data_id)
        # X = preprocessing_data_X(dataset.data.features)
    elif source == "csv":
        X = pd.read_csv(file_path).to_numpy()
    if X.shape[0]>2000:
        np.random.seed(seed)
        keep_idx = np.random.choice(X.shape[0], 2000, replace=False)
        X = X[keep_idx, :]
    if sample_row_n is not None:
        np.random.seed(seed)
        keep_idx = np.random.choice(X.shape[0], sample_row_n, replace=False)
        X = X[keep_idx, :]
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X
    

def sample_real_data_y(X=None, source=None, data_name=None, file_path=None, task_id=None, data_id=None,
                     seed=4307, sample_row_n=None, return_support=True):
    if source == "imodels":
        _, y, _ = imodels.get_clean_dataset(data_name)
    elif source == "openml":
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        _, y, _, _ = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
        if task_id == 361260 or task_id == 361622:
            y = np.log(y)
    elif source == "uci":
        dataset = fetch_ucirepo(id=data_id)
        # y = preprocessing_data_y(dataset.data.targets)
    elif source == "csv":
        y = pd.read_csv(file_path).to_numpy().flatten()
    if sample_row_n is not None:
        np.random.seed(seed)
        keep_idx = np.random.choice(y.shape[0], sample_row_n, replace=False)
        y = y[keep_idx]
    if y.shape[0]>2000:
        np.random.seed(seed)
        keep_idx = np.random.choice(y.shape[0], 2000, replace=False)
        y = y[keep_idx]
    if return_support:
        return y, np.ones(y.shape), None
    return y

X = sample_real_data_X(source="openml", task_id=9978)
y = sample_real_data_y(X, source="openml", task_id=9978)[0]
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# fit random forest with params from MDI+
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',min_samples_leaf=1,random_state=1)

print("------------------------")
print("First 5 Rows of X_train:")
print(X_train[:5])
print("------------------------")
print("First 5 Rows of y_train:")
print(y_train[:5])
print("------------------------")

# ridge rf+
rf_plus_ridge = RandomForestPlusClassifier(rf_model=rf,
                    prediction_model=LogisticRegressionCV(penalty='l2',
                                cv=5, max_iter=10000, random_state=0))
rf_plus_ridge.fit(X_train, y_train)

lmdi_explainer = RFPlusMDI(rf_plus_ridge, mode = "only_k", evaluate_on = "all")

lmdi_scores = np.abs(lmdi_explainer.explain_linear_partial(X_train, y_train, ranking=True))

train_fi_mean = np.mean(lmdi_scores, axis=0)
sorted_feature = np.argsort(-train_fi_mean)

print("------------------------")
print("Train Feature Importance Means:")
print(train_fi_mean)
print("------------------------")
print("Sorted Feature Importance:")
print(sorted_feature)
print("------------------------")
