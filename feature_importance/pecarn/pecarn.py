# imports
from methods import *
from sklearn.model_selection import train_test_split
import os

# get data
X, y = get_data()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf, rf_plus = fit_rf_models(X_train, y_train)

# shap
shap_values = get_shap(X_test, rf, is_boosting=False)

# lime
lime_values = get_lime(np.array(X_test), rf, is_boosting=False)

# local mdi
lmdi_values = local_mdi_score(np.array(X_train), np.array(X_test), model=rf)

# local mdi+
lmdi_plus_values = get_lmdi_plus(np.array(X_test), None, rf_plus)

# make results directory
curr_dir = os.getcwd()

# add results subdir
results_dir = os.path.join(curr_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# save to results subdir
np.savetxt(os.path.join(results_dir, 'shap.csv'), shap_values, delimiter=',')
np.savetxt(os.path.join(results_dir, 'lime.csv'), lime_values, delimiter=',')
np.savetxt(os.path.join(results_dir, 'lmdi.csv'), lmdi_values, delimiter=',')
np.savetxt(os.path.join(results_dir, 'lmdi_plus.csv'), lmdi_plus_values, delimiter=',')