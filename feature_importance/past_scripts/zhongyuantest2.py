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
import pickle

with open("local_feature_importances_low.pkl", "rb") as f: 
    local_feature_importances_low = pickle.load(f)

local_feature_importances = np.nanmean(local_feature_importances_low,axis=-1,dtype=np.float128)
print(local_feature_importances)