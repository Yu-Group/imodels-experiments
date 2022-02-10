import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import imodels
from imodels.util.data_util import get_clean_dataset

np.random.seed(42)
X, y, feature_names = get_clean_dataset('enhancer.csv', 'imodels')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for m, name in zip(
        [imodels.FIGSClassifier(max_rules=5), RandomForestClassifier()],
        ['FIGS', 'Random Forest']):
    m.fit(X_train, y_train)
    print(name)
    acc_test = accuracy_score(m.predict(X_test), y_test)
    acc_train = accuracy_score(m.predict(X_train), y_train)
    print(f'\ttest acc {acc_test:0.3f}')
    print(f'\ttrain acc {acc_train:0.3f}')
