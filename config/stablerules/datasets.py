import numpy as np

from util import Dataset

DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    # ("sonar", "sonar", "pmlb"),
    # ("heart", oj(DATASET_PATH, "heart.csv"), 'local'),
    # ("breast-cancer", oj(DATASET_PATH, "breast_cancer.csv"), 'local'),
    # ("haberman", oj(DATASET_PATH, "haberman.csv"), 'local'),
    # ("ionosphere", "ionosphere", 'pmlb'),
    # ("diabetes", "diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", oj(DATASET_PATH, "credit_g.csv"), 'local'), # like german-credit, but more feats
    # ("german-credit", "german", "pmlb"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    # our preprocessing in imodels-data places continuous columns in the front
    Dataset("juvenile_clean", 'imodels', 'juvenile', np.arange(3)),
    Dataset("compas_two_year_clean", 'imodels', 'recidivism', np.arange(7)),
    Dataset("credit_card_clean", 'imodels', 'credit', np.arange(20)),
    Dataset("readmission_clean", 'imodels', 'readmission', np.arange(10)),
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    # ('friedman1', 'friedman1', 'synthetic'),
    # ('friedman2', 'friedman2', 'synthetic'),
    # ('friedman3', 'friedman3', 'synthetic'),

    # ("diabetes-regr", "diabetes", 'sklearn'),
    # ("california-housing", "california_housing", 'sklearn'),
    # ("satellite-image", "294_satellite_image", 'pmlb'),
    # ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    # ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'), # this one is v big (100k examples)

]
