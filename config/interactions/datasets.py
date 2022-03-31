DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ("sonar", "sonar", "pmlb"),
    ("heart", "heart", 'imodels'),
    ("breast-cancer", "breast_cancer", 'imodels'),
    ("haberman", "haberman", 'imodels'),
    # ("ionosphere", "ionosphere", 'pmlb'),
    # ("diabetes", "diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
    # ("german-credit", "german", "pmlb"),

    # clinical-decision rules
    ("iai-pecarn", "iai_pecarn.csv", "imodels"),
    ("csi-pecarn", "csi_all.csv", "imodels"),
    ("tbi-pecarn", "tbi_pred.csv", "imodels"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", "juvenile_clean", 'imodels'),
    ("recidivism", "compas_two_year_clean", 'imodels'),
    # ("credit", "credit_card_clean", 'imodels'),
    # ("readmission", 'readmission_clean', 'imodels'),  # v big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),
    ("diabetes-regr", "diabetes", 'sklearn'),
    ('abalone', '183', 'openml'),    
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("satellite-image", "294_satellite_image", 'pmlb'),    
    ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues    
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)
]