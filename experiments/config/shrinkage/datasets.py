from os.path import join as oj

from experiments.util import DATASET_PATH
IMODELS_DATASET_PATH = oj(DATASET_PATH, 'imodels_data')

DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ("sonar", "sonar", "pmlb"),
    ("heart", oj(IMODELS_DATASET_PATH, "heart.csv"), 'local'),
    ("breast-cancer", oj(IMODELS_DATASET_PATH, "breast_cancer.csv"), 'local'),
    ("haberman", oj(IMODELS_DATASET_PATH, "haberman.csv"), 'local'),
    ("ionosphere", "ionosphere", 'pmlb'),
    ("diabetes", "diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", oj(DATASET_PATH, "credit_g.csv"), 'local'), # like german-credit, but more feats
    ("german-credit", "german", "pmlb"),

    # clinical-decision rules
    # ("iai-pecarn", oj(IMODELS_DATASET_PATH, "iai_pecarn.csv"), "local"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", oj(IMODELS_DATASET_PATH, "juvenile_clean.csv"), 'local'),
    ("recidivism", oj(IMODELS_DATASET_PATH, "compas_two_year_clean.csv"), 'local'),
    ("credit", oj(IMODELS_DATASET_PATH, "credit_card_clean.csv"), 'local'),
    ("readmission", oj(IMODELS_DATASET_PATH, 'readmission_clean.csv'), 'local'),  # v big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),
    ('abalone', '183', 'openml'),
    ("diabetes-regr", "diabetes", 'sklearn'),
    ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues
    ("satellite-image", "294_satellite_image", 'pmlb'),
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)

]
