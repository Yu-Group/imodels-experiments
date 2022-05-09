import os, sys
import pickle as pkl
import itertools
from functools import partial
from os.path import join as oj
from collections import defaultdict
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

import imodels
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imodels.util import data_util
from numpy import concatenate as npcat
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import metrics, model_selection
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(0)

# add project root to path
sys.path.append(oj(os.path.realpath(__file__).split('notebooks')[0]))

import validate

DATASET = 'iai'
RUN_VAL = False

if DATASET == 'iai':

    # much smaller proportion of positive class labels makes some random splits
    #   and spec94 metric too noisy to use
    seeds = [0, 1, 2, 4, 7, 11, 12, 13, 15, 16]
    VAL_METRICS = ['spec90']
else:
    seeds = range(10)
    VAL_METRICS = ['spec94', 'spec90']

max_leaf_nodes_options = [8, 12, 16]
tao_iter_options = [1, 5]

for seed in tqdm(seeds):
    SPLIT_SEED = seed
    RESULT_PATH = f'notebooks/transfertree/results_2/{DATASET}/seed_{SPLIT_SEED}'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    def all_stats_curve(y_test, preds_proba, plot=False, thresholds=None, model_name=None):
        '''preds_proba should be 1d
        '''
        if thresholds is None:
            thresholds = sorted(np.unique(preds_proba))
        all_stats = {
            s: [] for s in ['sens', 'spec', 'ppv', 'npv', 'lr+', 'lr-', 'f1']
        }
        for threshold in tqdm(thresholds):
            preds = preds_proba > threshold
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sens = tp / (tp + fn)
                spec = tn / (tn + fp)
                all_stats['sens'].append(sens)
                all_stats['spec'].append(spec)
                all_stats['ppv'].append(tp / (tp + fp))
                all_stats['npv'].append(tn / (tn + fn))
                all_stats['lr+'].append(sens / (1 - spec))
                all_stats['lr-'].append((1 - sens) / spec)
                all_stats['f1'].append(tp / (tp + 0.5 * (fp + fn)))

        if plot:
            if 'pecarn' in model_name.lower():
                plt.plot(all_stats['sens'][0], all_stats['spec'][0], '.-', label=model_name)
            else:
                plt.plot(all_stats['sens'], all_stats['spec'], '.-', label=model_name)
            plt.xlabel('sensitivity')
            plt.ylabel('specificity')
            plt.grid()
        return all_stats, thresholds


    results = defaultdict(lambda:[])
    columns = [f'spec9{i}' for i in range(0, 9, 2)] + ['aps', 'auc', 'acc', 'f1', 'args']


    def log_results(model, model_name, X_test, y_test, model_args=None, dct=None):
        pred_proba_args = (X_test,)
        
        spec_scorer_list = [validate.make_best_spec_high_sens_scorer(sens) for sens in [0.9, 0.92, 0.94, 0.96, 0.98]]
        spec_scores = [scorer(y_test, model.predict_proba(*pred_proba_args)[:, 1]) for scorer in spec_scorer_list]
        apc = metrics.average_precision_score(y_test, model.predict_proba(*pred_proba_args)[:, 1])
        auc = metrics.roc_auc_score(y_test, model.predict_proba(*pred_proba_args)[:, 1])
        acc = metrics.accuracy_score(y_test, model.predict(X_test))
        f1 = metrics.f1_score(y_test, model.predict(X_test))
        if dct is not None:
            dct[model_name] = spec_scores + [apc, auc, acc, f1, model_args]
        else:
            results[model_name] = spec_scores + [apc, auc, acc, f1, model_args]


    class TransferTree:
        def __init__(self, model_0, model_1, model_1_log_arr):
            self.model_0 = model_0
            self.model_1 = model_1
            self.model_1_log_arr = model_1_log_arr

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

        def predict_proba(self, X):
            preds_proba = np.zeros((X.shape[0], 2))
            preds_proba[~self.model_1_log_arr] = self.model_0.predict_proba(
                X[~self.model_1_log_arr])
            preds_proba[self.model_1_log_arr] = self.model_1.predict_proba(
                X[self.model_1_log_arr])
            return preds_proba


    class PECARNModel:
        def __init__(self, young):
            self.young = young

        def predict(self, X: pd.DataFrame):
            if DATASET == 'tbi' and self.young:
                factors_sum = (
                    X['AMS'] + X['HemaLoc_Occipital'] + X['HemaLoc_Parietal/Temporal'] + X['LocLen_1-5 min'] + 
                    X['LocLen_5 sec - 1 min'] + X['LocLen_>5 min'] + X['High_impact_InjSev_High'] + 
                    X['SFxPalp_Unclear'] + X['SFxPalp_Yes'] + (1 - X['ActNorm']))
            elif DATASET == 'tbi':
                factors_sum = (
                    X['AMS'] + X['Vomit'] + X['LOCSeparate_Suspected'] + X['LOCSeparate_Yes'] + 
                    X['High_impact_InjSev_High'] + X['SFxBas'] +  X['HASeverity_Severe'])
            elif DATASET == 'csi':
                factors_sum = (
                    X['AlteredMentalStatus2'] + X['PainNeck2'] + X['FocalNeuroFindings2'] + 
                    X['Torticollis2'] + X['subinj_TorsoTrunk2'] + X['Predisposed'] + 
                    X['HighriskDiving'] + X['HighriskMVC']
                )
            elif DATASET == 'iai':
                factors_sum = (
                    X['AbdTrauma_or_SeatBeltSign_yes'] + (X['GCSScore'] <= 13).astype(int) + 
                    X['AbdTenderDegree_Mild'] + X['AbdTenderDegree_Moderate'] + 
                    X['AbdTenderDegree_Severe'] + X['ThoracicTrauma_yes'] + X['AbdomenPain_yes'] +
                    X['DecrBreathSound_yes'] + X['VomitWretch_yes']
                )
            preds = (factors_sum >= 1).astype(int)

            return preds.values
        
        def predict_proba(self, X: pd.DataFrame):
            preds = np.expand_dims(self.predict(X), axis=1)
            return np.hstack((1 - preds, preds))


    X, y, feature_names = data_util.get_clean_dataset(f'{DATASET}_pecarn_pred.csv', data_source='imodels')
    X_df = pd.DataFrame(X, columns=feature_names)
    # X_df['Age<2_no'].value_counts()


    def predict_and_save(model, X_test, y_test, model_name, group):
        '''Plots cv and returns cv, saves all stats
        '''
        results = {}
        for x, y, suffix in zip([X_test],
                                [y_test],
                                ['_tune']):
            stats, threshes = all_stats_curve(y, model.predict_proba(x)[:, 1], plot=False, model_name=model_name)
            for stat in stats.keys():
                results[stat + suffix] = stats[stat]
            results['threshes' + suffix] = threshes
            results['acc'] = metrics.accuracy_score(y, model.predict(x))
            results['f1'] = metrics.f1_score(y, model.predict(x))
            if type(model) not in {TransferTree, PECARNModel}:
                results['params'] = model.get_params()
        if not os.path.exists(oj(RESULT_PATH, group)):
            os.mkdir(oj(RESULT_PATH, group))
        pkl.dump(results, open(oj(RESULT_PATH, group, model_name + '.pkl'), 'wb'))
        return stats, threshes


    ### training propensity model
    X_prop_raw, _, fnames_prop = data_util.get_clean_dataset(f'{DATASET}_pecarn_prop.csv', data_source='imodels', convertna=False)
    X_df_prop_raw = pd.DataFrame(X_prop_raw, columns=fnames_prop)
    X_df_prop_raw['outcome'] = y

    if DATASET == 'tbi':
        y_prop = X_df_prop_raw['AgeTwoPlus']
        X_df_prop = X_df_prop_raw.drop(columns=['AgeinYears', 'AgeInMonth', 'AgeTwoPlus', 'outcome'])
    elif DATASET == 'csi':
        y_prop = (X_df_prop_raw['AgeInYears'] >= 2).astype(int)
        X_df_prop = X_df_prop_raw.drop(columns=['AgeInYears', 'outcome'])
    elif DATASET == 'iai':
        y_prop = X_df_prop_raw['Age<2_no']
        X_df_prop = X_df_prop_raw.drop(columns=['Age', 'Age<2_no', 'Age<2_yes', 'outcome'])

    X_prop = X_df_prop.values
    X_prop_train_full, X_prop_test, y_prop_train_full, y_prop_test = model_selection.train_test_split(X_prop, y_prop, test_size=0.2, random_state=SPLIT_SEED)

    prop_models = {}
    prop_scores = {}
    if DATASET == 'csi':
        # much smaller val set so try not to overfit prop model
        prop_models['L'] = LogisticRegression(C=2.783, penalty='l2', solver='liblinear')
        prop_models['GBL'] = GradientBoostingClassifier()
    else:
        prop_models['LL'] = LogisticRegression(C=2.783, penalty='l2', solver='liblinear')
        prop_models['LS'] = LogisticRegression(C=0.01, penalty='l2', solver='liblinear')
        prop_models['GBL'] = GradientBoostingClassifier()
        prop_models['GBS'] = GradientBoostingClassifier(n_estimators=50)

    for m in prop_models:
        prop_models[m].fit(X_prop_train_full, y_prop_train_full)
        prop_scores[f'{m}_full'] = prop_models[m].predict_proba(X_prop_train_full)[:, 1]
        prop_scores[m] = model_selection.train_test_split(
            prop_scores[f'{m}_full'], test_size=0.25, random_state=SPLIT_SEED)[0]

    ### data setup
    if DATASET == 'csi':
        is_group_1 = (X_df['AgeInYears'] >= 2).astype(bool)
        X_df_clean = X_df
    elif DATASET == 'tbi':
        is_group_1 = X_df['AgeTwoPlus'].astype(bool)
        X_df_clean = X_df.drop(columns=['AgeinYears'])
    elif DATASET == 'iai':
        is_group_1 = X_df['Age<2_no'].astype(bool)
        X_df_clean = X_df#.drop(columns=['AgeinYears'])

    X_train_full, X_test, y_train_full, y_test, is_group_1_train_full, is_group_1_test = (
        model_selection.train_test_split(X_df_clean, y, is_group_1, test_size=0.2, random_state=SPLIT_SEED))
    X_train, X_val, y_train, y_val, is_group_1_train, is_group_1_val = (
        model_selection.train_test_split(X_train_full, y_train_full, is_group_1_train_full, test_size=0.25, random_state=SPLIT_SEED))

    X_train_young, X_val_young, X_test_young = X_train[~is_group_1_train], X_val[~is_group_1_val], X_test[~is_group_1_test]
    X_train_old, X_val_old, X_test_old = X_train[is_group_1_train], X_val[is_group_1_val], X_test[is_group_1_test]
    y_train_young, y_val_young, y_test_young = y_train[~is_group_1_train], y_val[~is_group_1_val], y_test[~is_group_1_test]
    y_train_old, y_val_old, y_test_old = y_train[is_group_1_train], y_val[is_group_1_val], y_test[is_group_1_test]

    X_train_full_old, X_train_full_young = pd.concat((X_train_old, X_val_old)), pd.concat((X_train_young, X_val_young))
    y_train_full_old, y_train_full_young = npcat((y_train_old, y_val_old)), npcat((y_train_young, y_val_young))
    cls_ratio = lambda x: int(pd.Series(x).value_counts()[0.0] / pd.Series(x).value_counts()[1.0])

    # print(f'seed {SPLIT_SEED} old', cls_ratio(y_test_old))
    # print(f'seed {SPLIT_SEED} young', cls_ratio(y_test_young))
    # print(f'seed {SPLIT_SEED} all val', cls_ratio(y_val))
    # print(f'seed {SPLIT_SEED} all test', cls_ratio(y_test))
    # continue

    # weight up positive examples due to high-sensitivity medical context
    cls_ratio_train_young = cls_ratio(y_train_young)
    cls_ratio_train_old = cls_ratio(y_train_old)
    cls_ratio_train = cls_ratio(y_train)

    sw_train_young = y_train_young * cls_ratio_train_young + 1
    sw_train_old = y_train_old * cls_ratio_train_old + 1
    sw_train = y_train * cls_ratio_train + 1

    sw_train_full_young = y_train_full_young * cls_ratio_train_young + 1
    sw_train_full_old = y_train_full_old * cls_ratio_train_old + 1
    sw_train_full = y_train_full * cls_ratio_train + 1


    def fit_models(model_cls, model_name, prop=False, tt=False, all=False):
        if tt:
            tao_iter_options_local = [0]
        elif model_cls == imodels.TaoTreeClassifier:
            tao_iter_options_local = tao_iter_options
        else:
            tao_iter_options_local = [None]
        
        if not prop:
            prop_models_local = [None]
        else:
            prop_models_local = prop_models

        for pmodel_name, msize, tao_iter in itertools.product(
            prop_models_local, max_leaf_nodes_options, tao_iter_options_local):
            if model_cls == DecisionTreeClassifier:
                model_args = {'max_leaf_nodes': msize, 'random_state': 0}
            elif model_cls == imodels.FIGSClassifier:
                model_args = {'max_rules': msize}
            elif model_cls == imodels.TaoTreeClassifier:
                model_args = {'n_iters': tao_iter, 'model_args': {'max_leaf_nodes': msize, 'random_state': 0},  
                    'update_scoring': 'average_precision'}

            young = model_cls(**model_args)
            old = model_cls(**model_args)
            whole = model_cls(**model_args)

            if not prop:
                young.fit(X_train_young, y_train_young, sample_weight=sw_train_young)
                old.fit(X_train_old, y_train_old, sample_weight=sw_train_old)
            else:
                young.fit(X_train, y_train, sample_weight=(1 - prop_scores[pmodel_name]) * sw_train)
                old.fit(X_train, y_train, sample_weight=prop_scores[pmodel_name] * sw_train)
            
            if tt:
                for _ in range(2):
                    num_updates = young._tao_iter_cart(
                        X_train_old.values, y_train_old, young.model.tree_, 
                        X_train_young.values, y_train_young, sample_weight=sw_train_old, sample_weight_score=sw_train_young)
                    if num_updates == 0:
                            break
            
                for _ in range(2):
                    num_updates = old._tao_iter_cart(
                        X_train_young.values, y_train_young, old.model.tree_, 
                        X_train_old.values, y_train_old, sample_weight=sw_train_young, sample_weight_score=sw_train_old)
                    if num_updates == 0:
                            break
                    
            name_young = f'{model_name}_<2_{msize}'
            name_old = f'{model_name}_>2_{msize}'
            name_whole = f'{model_name}_all_{msize}'

            if tao_iter not in [None, 0] and not tt:
                name_young += f'_{tao_iter}'
                name_old += f'_{tao_iter}'
                name_whole += f'_{tao_iter}'
            
            if prop:
                name_young = pmodel_name + name_young
                name_old = pmodel_name + name_old
                name_whole = pmodel_name + name_whole

            log_results(young, name_young, X_val_young, y_val_young, model_args)
            log_results(old, name_old, X_val_old, y_val_old, model_args)

            if all:
                whole.fit(X_train, y_train, sw_train)
                log_results(whole, name_whole, X_val, y_val, model_args)
    
    if RUN_VAL:
        # CART validation
        fit_models(DecisionTreeClassifier, 'CART', all=True)
        fit_models(DecisionTreeClassifier, 'PCART', prop=True)
        # # fit_models(imodels.TaoTreeClassifier, 'TTCART', tt=True)

        # FIGS validation
        fit_models(imodels.FIGSClassifier, 'FIGS', all=True)
        fit_models(imodels.FIGSClassifier, 'PFIGS', prop=True)
        
        # TAO validation
        fit_models(imodels.TaoTreeClassifier, 'TAO', all=True)

        val_df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
        val_df.to_csv(oj(RESULT_PATH, 'val.csv'))
    else:
        # Loading previously run validation
        val_df = pd.read_csv(oj(RESULT_PATH, 'val.csv')).set_index('Unnamed: 0')
        val_df['args'] = val_df['args'].apply(eval)

    args = val_df['args']
    best_models = {}

    def get_best_args(val_df_group, model_name, round=2):
        return val_df_group.filter(regex=model_name, axis=0).round(round).sort_values(
            by=VAL_METRICS, kind='mergesort', ascending=False)['args'].iloc[0]
    
    # results for simple all-data fits
    all_results = val_df[val_df.index.str.contains('all')]
    best_models['cart_all'] = DecisionTreeClassifier(**get_best_args(all_results, 'CART')).fit(X_train_full, y_train_full, sw_train_full)
    best_models['figs_all'] = imodels.FIGSClassifier(**get_best_args(all_results, 'FIGS')).fit(X_train_full, y_train_full, sample_weight=sw_train_full)
    best_models['tao_all'] = imodels.TaoTreeClassifier(**get_best_args(all_results, 'TAO')).fit(X_train_full, y_train_full, sample_weight=sw_train_full)
    
    # results for >2 group
    old_results = val_df[val_df.index.str.contains('>2')]
    best_models['cart_old'] = DecisionTreeClassifier(**get_best_args(old_results, '^CART')).fit(
        X_train_full_old, y_train_full_old, sw_train_full_old)
    best_models['figs_old'] = imodels.FIGSClassifier(**get_best_args(old_results, '^FIGS')).fit(
        X_train_full_old, y_train_full_old, sample_weight=sw_train_full_old)
    best_models['tao_old'] = imodels.TaoTreeClassifier(**get_best_args(old_results, '^TAO')).fit(
        X_train_full_old, y_train_full_old, sample_weight = sw_train_full_old)
    best_models['pecarn_old'] = PECARNModel(young=False)
    
    # results for <2 group
    young_results = val_df[val_df.index.str.contains('<2')]
    best_models['cart_young'] = DecisionTreeClassifier(**get_best_args(young_results, '^CART')).fit(
        X_train_full_young, y_train_full_young, sw_train_full_young)
    best_models['figs_young'] = imodels.FIGSClassifier(**get_best_args(young_results, '^FIGS')).fit(
        X_train_full_young, y_train_full_young, sample_weight=sw_train_full_young)
    best_models['tao_young'] = imodels.TaoTreeClassifier(**get_best_args(young_results, '^TAO')).fit(
        X_train_full_young, y_train_full_young, sample_weight = sw_train_full_young)
    best_models['pecarn_young'] = PECARNModel(young=True)

    # all ages results
    for model_name in ['pecarn', 'figs', 'tao', 'cart']:
        best_models[f'{model_name}_combine'] = TransferTree(
            best_models[f'{model_name}_young'], best_models[f'{model_name}_old'], is_group_1_test)

    # use validation to select best propensity model and most 
    results_pmodel = defaultdict(lambda:[])
    for model, cls in [('PFIGS', imodels.FIGSClassifier), ('PCART', DecisionTreeClassifier)]:
        model_parent = model[1:]
        for pmodel_name in prop_models:
            best_young_model = cls(**get_best_args(young_results, f'^{pmodel_name}{model}')).fit(
            X_train, y_train, sample_weight=(1 - prop_scores[pmodel_name]) * sw_train)
            best_old_model = cls(**get_best_args(old_results, f'^{pmodel_name}{model}')).fit(
            X_train, y_train, sample_weight=prop_scores[pmodel_name] * sw_train)

            combine = TransferTree(best_young_model, best_old_model, is_group_1_val)
            best_parent_train_only = cls(**get_best_args(all_results, f'^{model_parent}')).fit(
                X_train, y_train, sample_weight=sw_train)
            pmix_young = TransferTree(best_young_model, best_parent_train_only, is_group_1_val)
            pmix_old = TransferTree(best_parent_train_only, best_old_model, is_group_1_val)

            log_results(combine, f'{model}_{pmodel_name}_all', X_val, y_val, pmodel_name, results_pmodel)
            log_results(pmix_young, f'MIX{model}_{pmodel_name}_young', X_val, y_val, 'young', results_pmodel)
            log_results(pmix_old, f'MIX{model}_{pmodel_name}_old', X_val, y_val, 'old', results_pmodel)
        
        results_pmodel_df = pd.DataFrame.from_dict(results_pmodel, orient='index', columns=columns)
        best_pmodel = get_best_args(results_pmodel_df, f'^{model}')
        best_pmix = get_best_args(results_pmodel_df, f'^MIX{model}_{best_pmodel}')

        if model == 'PFIGS':
            print(results_pmodel_df)
            print(best_pmodel)
            print(best_pmix)

        best_young_model_final = cls(**get_best_args(young_results, f'^{best_pmodel}{model}')).fit(
            X_train_full, y_train_full, sample_weight=(1 - prop_scores[f'{best_pmodel}_full']) * sw_train_full)
        best_old_model_final = cls(**get_best_args(old_results, f'^{best_pmodel}{model}')).fit(
            X_train_full, y_train_full, sample_weight=prop_scores[f'{best_pmodel}_full'] * sw_train_full)

        best_models[f'{model}_young'.lower()] = best_young_model_final
        best_models[f'{model}_old'.lower()] = best_old_model_final
        best_models[f'{model}_combine'.lower()] = TransferTree(
                best_young_model_final, best_old_model_final, is_group_1_test)
            
        if best_pmix == 'young':
            best_models[f'{model}_mix'.lower()] = TransferTree(
                best_models[f'{model}_young'.lower()], best_models[f'{model_parent}_all'.lower()], is_group_1_test)
        else:
            best_models[f'{model}_mix'.lower()] = TransferTree(
                best_models[f'{model_parent}_all'.lower()], best_models[f'{model}_old'.lower()], is_group_1_test)

    results_pmodel_df.to_csv(oj(RESULT_PATH, 'pmodel_val.csv'))

    # use validation to select the best mix 
    results_pmix = defaultdict(lambda:[])
    for model, cls in [
        ('FIGS', imodels.FIGSClassifier), ('CART', DecisionTreeClassifier), ('TAO', imodels.TaoTreeClassifier)]:

        best_young_model = cls(**get_best_args(young_results, f'^{model}')).fit(
            X_train_young, y_train_young, sample_weight=sw_train_young)
        best_old_model = cls(**get_best_args(old_results, f'^{model}')).fit(
            X_train_old, y_train_old, sample_weight=sw_train_old)

        best_parent_train_only = cls(**get_best_args(all_results, f'^{model}')).fit(
            X_train, y_train, sample_weight=sw_train)
        pmix_young = TransferTree(best_young_model, best_parent_train_only, is_group_1_val)
        pmix_old = TransferTree(best_parent_train_only, best_old_model, is_group_1_val)
    
        log_results(pmix_young, f'MIX{model}_young', X_val, y_val, 'young', results_pmix)
        log_results(pmix_old, f'MIX{model}_old', X_val, y_val, 'old', results_pmix)
        
        results_pmix_df = pd.DataFrame.from_dict(results_pmix, orient='index', columns=columns)
        best_pmix = get_best_args(results_pmix_df, f'^MIX{model}')

        if best_pmix == 'young':
            best_models[f'{model}_mix'.lower()] = TransferTree(
                best_models[f'{model}_young'.lower()], best_models[f'{model}_all'.lower()], is_group_1_test)
        else:
            best_models[f'{model}_mix'.lower()] = TransferTree(
                best_models[f'{model}_all'.lower()], best_models[f'{model}_old'.lower()], is_group_1_test)

    results_pmix_df.to_csv(oj(RESULT_PATH, 'pmix_val.csv'))
    
    for model_name in [
        'cart_all', 'cart_combine', 'cart_mix', 
        'tao_all', 'tao_combine', 'tao_mix',
        'figs_all', 'figs_combine', 'figs_mix', 
        'pcart_combine', 'pcart_mix', 'pfigs_combine', 'pfigs_mix', 'pecarn_combine']:
        predict_and_save(best_models[f'{model_name}'], X_test, y_test, f'{model_name}', 'all')

    for model_name in [
        'cart_old', 'pcart_old', 'pfigs_old', 'figs_old', 'tao_old', 'pecarn_old', 'cart_all', 'figs_all', 'tao_all']:
        predict_and_save(best_models[f'{model_name}'], X_test_old, y_test_old, f'{model_name}', 'old')

    for model_name in [
        'cart_young', 'pcart_young', 'figs_young', 'pfigs_young', 'tao_young', 'pecarn_young', 'cart_all', 'figs_all', 'tao_all']:
        predict_and_save(best_models[f'{model_name}'], X_test_young, y_test_young, f'{model_name}', 'young')

    pkl.dump(best_models, open(oj(RESULT_PATH, 'best_models.pkl'), 'wb'))
