from util import Dataset

DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    Dataset("sonar", "pmlb"),
    Dataset("heart", 'imodels'),
    Dataset("breast_cancer", 'imodels', 'breast-cancer'),
    Dataset("haberman", 'imodels'),
    Dataset("ionosphere", 'pmlb'),
    Dataset("diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
    Dataset("german", "pmlb", "german-credit"),

    # clinical-decision rules
    # ("iai-pecarn", "iai_pecarn.csv", "imodels"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    Dataset("juvenile_clean", 'imodels', 'juvenile'),
    Dataset("compas_two_year_clean", 'imodels', 'recidivism'),
    Dataset("credit_card_clean", 'imodels', 'credit'),
    Dataset("readmission_clean", 'imodels', 'readmission'),  # big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    Dataset('friedman1', 'synthetic'),
    Dataset('friedman2', 'synthetic'),
    Dataset('friedman3', 'synthetic'),
    Dataset('183', 'openml', 'abalone',),
    Dataset("diabetes", 'sklearn', "diabetes-regr"),
    Dataset("california_housing", 'sklearn', "california-housing"),  # this replaced boston-housing due to ethical issues
    Dataset("294_satellite_image", 'pmlb', "satellite-image"),
    Dataset("1199_BNG_echoMonths", 'pmlb', "echo-months",),
    Dataset("1201_BNG_breastTumor", 'pmlb', "breast-tumor",),  # this one is v big (100k examples)
]
