import pandas as pd
import numpy as np
import os

import pyreadstat # loading .sav 

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def generate_regr_data(data, covariates_continuous, covariates_categorical, objective):
    '''
    Generate data X, y and name_covariates from data
    covariates_continuous: Names of continuous feature column
    covariates_categorical: Names of categorical feature column
    objective: Name of the objective column
    All the rows containing NaN are discarded
    NOTE: Categorcal First.
    '''
    data = data[covariates_continuous + covariates_categorical + [objective]]
    data = data.dropna()

    if covariates_categorical != []:
        X_categorical = data[covariates_categorical].astype('str')
        X_categorical = pd.get_dummies(X_categorical, drop_first=True)
        names_categorical = np.array(X_categorical.columns)
        X_categorical = np.array(X_categorical)
    if covariates_continuous != []:
        X_continuous = np.array(data[covariates_continuous])
        # Normalize
        X_continuous = StandardScaler().fit_transform(X_continuous)
    # Concatenate categorical and continuous covariates  
    if covariates_categorical == []:
        X = X_continuous
        names_covariates = covariates_continuous
    elif covariates_continuous == []:
        X = X_categorical
        names_covariates = names_categorical
    else:
        X = np.concatenate([X_categorical, X_continuous], axis=1)
        names_covariates = np.concatenate([names_categorical, covariates_continuous])

    y = np.array(data[objective])

    return X, y, names_covariates


def load_regr_data(name_data, dir_data='../data/regr'):
    if 'Dutch_drinking' in name_data:
        # Adolescent Heavy Drinking Does Not Affect Maturation of Basic Executive 
        # Functioning: Longitudinal Findings from the TRAILS Study
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139186#pone-0139186-t005
        # Table 5
        
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'Dutch_drinking.sav'))
        
        category = name_data[15:]

        covariates_categorical = ['Imputation_']
        covariates_continuous = ['t1'+category, 'sex', 't1age', 't1ses', 't1mat_alcohol', 't1pat_alcohol',
                                 't1ysr_del', 't3year_cannabis', 't4year_cannabis', 
                                 't3daily_smoking', 't4month_smoking']
        objective = 'Zdelta_' + category

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'Brazil_health' in name_data:
        # Did the Family Health Strategy have an impact on indicators of hospitalizations 
        # for stroke and heart failure? Longitudinal study in Brazil: 1998-2013
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198428#sec011
        # Table 3 (Heart) Table 4 (Stroke)
        
        data = pd.read_excel(os.path.join(dir_data, 'Brazil_health.xls'))
        
        data['ESFProportion'] = [str(s).replace(',', '.').replace('..', '.') for s in list(data['ESFProportion'])]
        data['ESFProportion'] = data['ESFProportion'].astype('float')

        data = data.loc[data['GDP']!='.']
        
        covariates_categorical = []
        covariates_continuous = ['Year', 'ESFProportion', 'ACSProportion', 'Population', 'GDP', 'DHI Value']
        if 'heart' in name_data:
            objective = 'HeartFailure per 10000'
            data = data.loc[data[objective]!='.']
        elif 'stroke' in name_data:
            objective = 'Stroke per 10000'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'Korea_grip' in name_data:
        # Association between grip strength and hand and knee radiographic osteoarthritis in 
        # Korean adults: Data from the Dong-gu study
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0185343#sec017
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'Korea_grip.xlsx'))

        if 'women' in name_data:
            data = data.loc[data['sex']==2]
        else:
            data = data.loc[data['sex']==1]
            
        covariates_categorical = []
        covariates_continuous = ['total_s_hand', 'JSN_hand', 'OP_hand', 'total_s_knee', 
                                 'OP_knee', 'JSN_knee',
                                 'age', 'BMI', 'smoking_c', 'alcohol_c', 'edu_2']
        objective = 'grip_strength'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'China_glucose' in name_data:
        # Fasting plasma glucose and serum uric acid levels in a general Chinese 
        # population with normal glucose tolerance: A U-shaped curve
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180111#sec016
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'China_glucose.xlsx'))
        
        if 'women1' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==2)&(data['FPG']<4.6)]
        elif 'women2' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==2)&(data['FPG']>=4.6)]
        elif 'men1' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==1)&(data['FPG']<4.7)]
        elif 'men2' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==1)&(data['FPG']>=4.7)]
            
        covariates_categorical = []
        covariates_continuous = ['FPG', 'Age', 'BMI', 'SBP', 'DBP', 'TC', 'TG', 'Drinker(N0/Y1)',
                                 'Smoker(N0/Y1)', 'eGFR', 'INS']
        objective = 'SUA'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
    
    elif name_data == 'Spain_Hair':
        # Hair cortisol concentrations in a Spanish sample of healthy adults
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204807
        # Table 3
        # Data: https://figshare.com/s/c27f4958b81b188dab4e
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'Spain_Hair_Healthy.sav'))
        
        
        covariates_categorical = []
        covariates_continuous = ['Age', 'Education', 'EmploymentS', 'HairDye', 'PhysicalAct']
        objective = 'Logcortisol'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, covariates_categorical, objective)
        
    elif name_data == 'China_HIV':
        # Stigma against People Living with HIV/AIDS in China: Does the Route of Infection Matter?
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0151078#sec014
        # Table 2 model b
        
        data = pd.read_stata(os.path.join(dir_data, 'China_HIV.dta'))
        
        # self-esteem: se???
        # Model b
        covariates_categorical = ['route', 'sex', 'ethni', 'relig', 'residence', 'marital', 'income',
                                  'coinf', 'smk', 'alch', 'drug', 'depression']
        covariates_continuous = ['yschool', 'age', 'resi', 'cope', 'ssupp', 'anxitot', 'se']
        # Model c
        # covariates_categorical = ['route', 'sex', 'ethni', 'relig', 'residence', 'marital', 'income', 'coinf']
        # covariates_continuous = ['yschool', 'age']
        objective = 'stigma'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)

    return X, y, names_covariates

