import numpy as np
import pandas as pd
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  AloRFPlusMDI, RFPlusMDI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from imodels import get_clean_dataset
from sklearn.linear_model import RidgeCV, LassoCV
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

# np.set_printoptions(precision=8)
# np.random.seed(42)  # Ensure consistent randomness across machines


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

X = sample_real_data_X(source="openml", task_id=361251)
y = sample_real_data_y(X, source="openml", task_id=361251)[0]

# Cast arrays to float32 globally
X = X.astype(np.float32)
y = y.astype(np.float32)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape)
est = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=1)
est.fit(X_train, y_train)

rf_plus_ridge = RandomForestPlusRegressor(rf_model=est, prediction_model=LassoCV(cv=5, max_iter=10000, random_state=0))
rf_plus_ridge.fit(X_train, y_train)

lmdi_explainer = RFPlusMDI(rf_plus_ridge, mode = "only_k", evaluate_on = "all")

lmdi_scores = np.abs(lmdi_explainer.explain_linear_partial(X_train, y_train))

print(lmdi_scores)
train_fi_mean = np.mean(lmdi_scores, axis=0)
print(train_fi_mean)
sorted_feature = np.argsort(-train_fi_mean)
print(sorted_feature)