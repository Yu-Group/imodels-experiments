import copy
import os
from os.path import join as oj
import glob
import argparse
import pickle as pkl
import time
import warnings
from scipy import stats
import dask
from dask.distributed import Client
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from collections import defaultdict
from typing import Callable, List, Tuple
import itertools
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, recall_score, precision_score, mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from sklearn.linear_model import Ridge
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import fi_config
from util import ModelConfig, FIModelConfig, tp, fp, neg, pos, specificity_score, auroc_score, auprc_score, compute_nsg_feat_corr_w_sig_subspace, apply_splitting_strategy
import dill
warnings.filterwarnings("ignore", message="Bins whose width")

#RUN THE FILE
def rbo(s, t, p, k=None, side="top", uneven_lengths=False):
    """
    Rank-based overlap (RBO) metric.
    Proposed in "A Similarity Measure for Indefinite Rankings" by
    Webber et al. (2010). This is the python implementation
    of gespeR::rbo from R.

    Parameters
    ----------
    s: array-like of shape (n,)
        Numeric vector.
    t: array-like of shape (n,)
        Numeric vector of same length as s.
    p: float between 0 and 1
        Weighting parameter in [0, 1]. High p implies strong emphasis
        on the top-ranked elements (i.e, the larger elements).
    k: None or int
        Evaluation depth for extrapolation
    side: string in {"top", "bottom"}
        Evaluate similarity between the top or the bottom of the
        ranked lists.
    uneven_lengths: bool
        Indicator if lists have uneven lengths.

    Returns
    -------
    Scalar value between 0 and 1, quantifying how much the
    rankings of x and y agree with each other. A higher
    values indicates greater similarity.

    """
    assert side in ["top", "bottom"]
    if k is None:
        k = int(np.floor(max(len(s), len(t)) / 2))
    if side == "top":
        ids = {"s": _select_ids(s, "top"),
               "t": _select_ids(t, "top")}
    elif side == "bottom":
        ids = {"s": _select_ids(s, "bottom"),
               "t": _select_ids(t, "bottom")}
    return min(1, _rbo_ext(ids["s"], ids["t"], p, k, uneven_lengths=uneven_lengths))


def _select_ids(x, side="top"):
    assert side in ["top", "bottom"]
    if side == "top":
        return np.argsort(-x)
    elif side == "bottom":
        return np.argsort(x)


def _rbo_ext(x, y, p, k, uneven_lengths=True):
    if len(x) <= len(y):
        S = x
        L = y
    else:
        S = y
        L = x
    l = min(k, len(L))
    s = min(k, len(S))
    if uneven_lengths:
        Xd = [len(np.intersect1d(S[:(i+1)], L[:(i+1)])) for i in range(l)]
        if l > s:
            sl_range = np.arange(s+1, l+1)
        else:
            sl_range = np.arange(l, s+2)
        result = ((1 - p) / p) * \
                 ((sum(Xd[:l] / np.arange(1, l+1) * p**np.arange(1, l+1))) +
                  (sum(Xd[s-1] * (sl_range - s) / (s * sl_range) * p**sl_range))) + \
                 ((Xd[l-1] - Xd[s-1]) / l + (Xd[s-1] / s)) * p**l
    else:
        k = min(s, k)
        Xd = [len(np.intersect1d(x[:(i+1)], y[:(i+1)])) for i in range(k)]
        Xk = Xd[k-1]
        result = (Xk / k) * p**k + (((1 - p) / p) * sum((Xd / np.arange(1, k+1)) * p**np.arange(1, k+1)))
    return result

# def rank_biased_overlap(list1, list2, p=0.9):
#     """
#     Compute the Rank-Biased Overlap (RBO) between two ranked lists.

#     Parameters:
#     - list1: numpy array or list of the first ranked list
#     - list2: numpy array or list of the second ranked list
#     - p: the discount factor (default is 0.9, which is commonly used)

#     Returns:
#     - rbo: the Rank-Biased Overlap score
#     """

#     # Convert lists to numpy arrays if they're not already
#     list1 = np.asarray(list1)
#     list2 = np.asarray(list2)

#     # Get the indices that would sort the arrays in descending order
#     sorted_indices1 = np.argsort(-list1)
#     sorted_indices2 = np.argsort(-list2)

#     # Rank lists based on sorted indices
#     ranked_list1 = sorted_indices1
#     ranked_list2 = sorted_indices2

#     # Initialize the overlap
#     overlap = 0.0
#     min_len = min(len(ranked_list1), len(ranked_list2))
    
#     # Compute the RBO
#     for i in range(min_len):
#         # Calculate the overlap at rank i
#         rank_i_overlap = len(set(ranked_list1[:i+1]) & set(ranked_list2[:i+1]))
        
#         # Add the discounted overlap to the total
#         overlap += (rank_i_overlap / (i + 1)) * (p ** (i + 1))
    
#     # Normalize the score
#     normalization = (1 - p) / (1 - p ** (min_len + 1))
#     rbo = overlap * normalization
    
#     return rbo

def compare_estimators(estimators: List[ModelConfig],
                       fi_estimators: List[FIModelConfig],
                       X, y, support: List,
                       metrics: List[Tuple[str, Callable]],
                       args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, datasets, and metrics.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    # initialize results
    results = defaultdict(lambda: [])
    feature_importance_list = {"absolute": {}}

    # loop over model estimators
    for model in estimators:
        est = model.cls(**model.kwargs)

        # get kwargs for all fi_ests
        fi_kwargs = {}
        for fi_est in fi_estimators:
            fi_kwargs.update(fi_est.kwargs)

        # get groups of estimators for each splitting strategy
        fi_ests_dict = defaultdict(list)
        for fi_est in fi_estimators:
            fi_ests_dict[fi_est.splitting_strategy].append(fi_est)

        # loop over splitting strategies
        for splitting_strategy, fi_ests in fi_ests_dict.items():
            # implement provided splitting strategy
            if splitting_strategy is not None:
                X_train, X_tune, X_test, y_train, y_tune, y_test = apply_splitting_strategy(X, y, splitting_strategy, args.split_seed)
            else:
                X_train = X
                X_test = X
                y_train = y
                y_test = y
            
            # check if there are NA values in the data
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                print(X_train)
                print(y_train)
                raise ValueError("There are NA values in the data")
            if np.isnan(X_test).any() or np.isnan(y_test).any():
                print(X_test)
                print(y_test)
                raise ValueError("There are NA values in the data")

            #####
            print(X_train)
            print(y_train)
            print(X_test)
            print(y_test)
            print(y_train.mean())


            if args.fit_model:
                print("Fitting Models")
                # fit RF model
                start_rf = time.time()
                est.fit(X_train, y_train)
                end_rf = time.time()

                # fit default RF_plus model
                start_rf_plus = time.time()
                rf_plus_base = RandomForestPlusRegressor(rf_model=est)
                rf_plus_base.fit(X_train, y_train)
                end_rf_plus = time.time()

                # fit oob RF_plus model
                start_rf_plus_oob = time.time()
                rf_plus_base_oob = RandomForestPlusRegressor(rf_model=est, fit_on="oob")
                rf_plus_base_oob.fit(X_train, y_train)
                end_rf_plus_oob = time.time()

                #fit inbag RF_plus model
                start_rf_plus_inbag = time.time()
                rf_plus_base_inbag = RandomForestPlusRegressor(rf_model=est, include_raw=False, fit_on="inbag", prediction_model=Ridge(alpha=1e-6))
                rf_plus_base_inbag.fit(X_train, y_train)
                end_rf_plus_inbag = time.time()

                # get test results
                # test_all_mse_rf = mean_squared_error(y_test, est.predict(X_test))
                # test_all_r2_rf = r2_score(y_test, est.predict(X_test))
                # test_all_mse_rf_plus = mean_squared_error(y_test, rf_plus_base.predict(X_test))
                # test_all_r2_rf_plus = r2_score(y_test, rf_plus_base.predict(X_test))
                # test_all_mse_rf_plus_oob = mean_squared_error(y_test, rf_plus_base_oob.predict(X_test))
                # test_all_r2_rf_plus_oob = r2_score(y_test, rf_plus_base_oob.predict(X_test))
                # test_all_mse_rf_plus_inbag = mean_squared_error(y_test, rf_plus_base_inbag.predict(X_test))
                # test_all_r2_rf_plus_inbag = r2_score(y_test, rf_plus_base_inbag.predict(X_test))

                # fitted_results = {
                #         "Model": ["RF", "RF_plus", "RF_plus_oob", "RF_plus_inbag"],
                #         "MSE": [test_all_mse_rf, test_all_mse_rf_plus, test_all_mse_rf_plus_oob, test_all_mse_rf_plus_inbag],
                #         "R2": [test_all_r2_rf, test_all_r2_rf_plus, test_all_r2_rf_plus_oob, test_all_r2_rf_plus_inbag],
                #         "Time": [end_rf - start_rf, end_rf_plus - start_rf_plus, end_rf_plus_oob - start_rf_plus_oob, end_rf_plus_inbag - start_rf_plus_inbag]
                # }
                
                # os.makedirs(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}", exist_ok=True)
                # results_df = pd.DataFrame(fitted_results)
                # results_df.to_csv(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_fitted_summary_{args.split_seed}.csv", index=False)
                            

                # pickle_file = f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RF_{args.split_seed}.dill"
                # with open(pickle_file, 'wb') as file:
                #     dill.dump(est, file)
                # pickle_file = f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_default_{args.split_seed}.dill"
                # with open(pickle_file, 'wb') as file:
                #     dill.dump(rf_plus_base, file)
                # pickle_file = f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_oob_{args.split_seed}.dill"
                # with open(pickle_file, 'wb') as file:
                #     dill.dump(rf_plus_base_oob, file)
                # pickle_file = f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_inbag_{args.split_seed}.dill"
                # with open(pickle_file, 'wb') as file:
                #     dill.dump(rf_plus_base_inbag, file)
            
            np.random.seed(42)
            if X_train.shape[0] > 100:
                indices_train = np.random.choice(X_train.shape[0], 100, replace=False)
                X_train_subset = X_train[indices_train]
                y_train_subset = y_train[indices_train]
            else:
                indices_train = np.arange(X_train.shape[0])
                X_train_subset = X_train
                y_train_subset = y_train
            
            if X_test.shape[0] > 100:
                indices_test = np.random.choice(X_test.shape[0], 100, replace=False)
                X_test_subset = X_test[indices_test]
                y_test_subset = y_test[indices_test]
            else:
                indices_test = np.arange(X_test.shape[0])
                X_test_subset = X_test
                y_test_subset = y_test

            # loop over fi estimators
            for fi_est in tqdm(fi_ests):
                metric_results = {
                    'model': model.name,
                    'fi': fi_est.name,
                    'train_size': X_train.shape[0],
                    'train_subset_size': X_train_subset.shape[0],
                    'test_size': X_test.shape[0],
                    'test_subset_size': X_test_subset.shape[0],
                    'num_features': X_train.shape[1],
                    'data_split_seed': args.split_seed,
                }
                for i in range(X_train_subset.shape[0]):
                    metric_results[f'sample_train_{i}'] = indices_train[i]
                for i in range(X_test_subset.shape[0]):
                    metric_results[f'sample_test_{i}'] = indices_test[i]

                print("Load Models")
                start = time.time()
                # with open(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_default_{args.split_seed}.dill", 'rb') as file:
                #     rf_plus_base = dill.load(file)
                # if fi_est.base_model == "None":
                #     loaded_model = None
                # elif fi_est.base_model == "RF":
                #     with open(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RF_{args.split_seed}.dill", 'rb') as file:
                #         loaded_model = dill.load(file)
                # elif fi_est.base_model == "RFPlus_oob":
                #     with open(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_oob_{args.split_seed}.dill", 'rb') as file:
                #         loaded_model = dill.load(file)
                # elif fi_est.base_model == "RFPlus_inbag":
                #     with open(f"/scratch/users/zhongyuan_liang/saved_models/auroc/{args.folder_name}/RFPlus_inbag_{args.split_seed}.dill", 'rb') as file:
                #         loaded_model = dill.load(file)
                # elif fi_est.base_model == "RFPlus_default":
                #     loaded_model = rf_plus_base
                rf_plus_base = rf_plus_base
                if fi_est.base_model == "None":
                    loaded_model = None
                elif fi_est.base_model == "RF":
                   loaded_model = est
                elif fi_est.base_model == "RFPlus_oob":
                    loaded_model = rf_plus_base_oob
                elif fi_est.base_model == "RFPlus_inbag":
                    loaded_model = rf_plus_base_inbag
                elif fi_est.base_model == "RFPlus_default":
                    loaded_model = rf_plus_base
                end = time.time()
                metric_results['load_model_time'] = end - start
                print(f"done with loading models: {end - start}")

                local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset = fi_est.cls(X_train=X_train, y_train=y_train, X_train_subset = X_train_subset, y_train_subset=y_train_subset,
                                                                                                                                X_test=X_test, y_test=y_test, X_test_subset=X_test_subset, y_test_subset=y_test_subset,
                                                                                                                                fit=loaded_model, mode="absolute")
                if fi_est.name.startswith("Local_MDI+"):
                    local_fi_score_train_subset = local_fi_score_train[indices_train]
                feature_importance_list["absolute"][fi_est.name] = [local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset]
                
                all_data = {"train_subset": local_fi_score_train_subset, "test_subset": local_fi_score_test_subset,
                            "test": local_fi_score_test}
                all_x = {"train_subset": np.abs(X_train_subset), "test_subset": np.abs(X_test_subset),
                            "test": np.abs(X_test)}
                # for d in all_data:
                #     data = all_data[d]
                #     if not isinstance(data, np.ndarray):
                #         metric_results[f'auroc_{d}'] = None
                #         metric_results[f'auprc_{d}'] = None
                #     else:
                #         auroc = []
                #         auprc = []
                #         for i in range(data.shape[0]):
                #             if fi_est.ascending:
                #                 auroc.append(roc_auc_score(support, data[i]))
                #                 auprc.append(average_precision_score(support, data[i]))
                #             else:
                #                 auroc.append(roc_auc_score(support, -1*data[i]))
                #                 auprc.append(average_precision_score(support, -1*data[i]))
                #         metric_results[f'auroc_{d}'] = np.array(auroc).mean()
                #         metric_results[f'auprc_{d}'] = np.array(auprc).mean()
                #     print(f"{fi_est.name} done with {d} data with auroc: {metric_results[f'auroc_{d}']} and auprc: {metric_results[f'auprc_{d}']}")
                for d in all_data:
                    data = all_data[d]
                    if not isinstance(data, np.ndarray):
                        metric_results[f'rbo_{d}'] = None
                        metric_results[f'rbo_{d}'] = None
                    else:
                        support_d = all_x[d]
                        support_d[:, -5:] = 0
                        rbo_lst = []
                        for i in range(data.shape[0]):
                            # if fi_est.ascending:
                            #     rbo_lst.append(rbo.RankingSimilarity(support_d[i], data[i]).rbo())
                            # else:
                            #     rbo_lst.append(rbo.RankingSimilarity(support_d[i], -1*data[i]).rbo())
                            if fi_est.ascending:
                                rbo_lst.append(rbo(support_d[i], data[i], 0.9))
                            else:
                                rbo_lst.append(rbo(support_d[i], -1*data[i], 0.9))  
                        metric_results[f'rbo_{d}'] = np.array(rbo_lst).mean()
                    print(f"{fi_est.name} done with {d} data with rbo: {metric_results[f'rbo_{d}']}")
                # initialize results with metadata and metric results
                kwargs: dict = model.kwargs  # dict
                for k in kwargs:
                    results[k].append(kwargs[k])
                for k in fi_kwargs:
                    if k in fi_est.kwargs:
                        results[k].append(str(fi_est.kwargs[k]))
                    else:
                        results[k].append(None)
                for met_name, met_val in metric_results.items():
                    results[met_name].append(met_val)
    return results, feature_importance_list


def run_comparison(path: str,
                   X, y, support: List,
                   metrics: List[Tuple[str, Callable]],
                   estimators: List[ModelConfig],
                   fi_estimators: List[FIModelConfig],
                   args):
    estimator_name = estimators[0].name.split(' - ')[0]
    fi_estimators_all = [fi_estimator for fi_estimator in itertools.chain(*fi_estimators) \
                         if fi_estimator.model_type in estimators[0].model_type]
    model_comparison_files_all = [oj(path, f'{estimator_name}_{fi_estimator.name}_comparisons.pkl') \
                                  for fi_estimator in fi_estimators_all]
    
    feature_importance_all = oj(path, f'feature_importance.pkl')


    if args.parallel_id is not None:
        model_comparison_files_all = [f'_{args.parallel_id[0]}.'.join(model_comparison_file.split('.')) \
                                      for model_comparison_file in model_comparison_files_all]

    fi_estimators = []
    model_comparison_files = []
    for model_comparison_file, fi_estimator in zip(model_comparison_files_all, fi_estimators_all):
        if os.path.isfile(model_comparison_file) and not args.ignore_cache:
            print(
                f'{estimator_name} with {fi_estimator.name} results already computed and cached. use --ignore_cache to recompute')
        else:
            fi_estimators.append(fi_estimator)
            model_comparison_files.append(model_comparison_file)

    if len(fi_estimators) == 0:
        return

    results, fi_lst = compare_estimators(estimators=estimators,
                                 fi_estimators=fi_estimators,
                                 X=X, y=y, support=support,
                                 metrics=metrics,
                                 args=args)

    estimators_list = [e.name for e in estimators]
    metrics_list = [m[0] for m in metrics]

    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed
    if args.nosave_cols is not None:
        nosave_cols = np.unique([x.strip() for x in args.nosave_cols.split(",")])
    else:
        nosave_cols = []
    for col in nosave_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    pkl.dump(fi_lst, open(feature_importance_all, 'wb'))

    for model_comparison_file, fi_estimator in zip(model_comparison_files, fi_estimators):
        output_dict = {
            # metadata
            'sim_name': args.config,
            'estimators': estimators_list,
            'fi_estimators': fi_estimator.name,
            'metrics': metrics_list,

            # actual values
            'df': df.loc[df.fi == fi_estimator.name],
        }
        pkl.dump(output_dict, open(model_comparison_file, 'wb'))
    return df


def get_metrics():
    return [('rocauc', auroc_score), ('prauc', auprc_score)]


def reformat_results(results):
    results = results.reset_index().drop(columns=['index'])
    # fi_scores = pd.concat(results.pop('fi_scores').to_dict()). \
    #     reset_index(level=0).rename(columns={'level_0': 'index'})
    # results_df = pd.merge(results, fi_scores, left_index=True, right_on="index")
    # return results_df
    return results

def run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, fi_ests, metrics, args):
    os.makedirs(oj(path, val_name, "rep" + str(i)), exist_ok=True)
    #np.random.seed(i)
    max_iter = 100
    iter = 0
    while iter <= max_iter:  # regenerate data if y is constant
        X = X_dgp(**X_params_dict)
        y, support, beta = y_dgp(X, **y_params_dict, seed = i, return_support=True)
        if not all(y == y[0]):
            break
        iter += 1
    if iter > max_iter:
        raise ValueError("Response y is constant.")
    if args.omit_vars is not None:
        omit_vars = np.unique([int(x.strip()) for x in args.omit_vars.split(",")])
        support = np.delete(support, omit_vars)
        X = np.delete(X, omit_vars, axis=1)
        del beta  # note: beta is not currently supported when using omit_vars

    for est in ests:
        results = run_comparison(path=oj(path, val_name, "rep" + str(i)),
                                 X=X, y=y, support=support,
                                 metrics=metrics,
                                 estimators=est,
                                 fi_estimators=fi_ests,
                                 args=args)
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    default_dir = os.getenv("SCRATCH")
    if default_dir is not None:
        default_dir = oj(default_dir, "feature_importance", "results")
    else:
        default_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results')

    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--model', type=str, default=None)  # , default='c4')
    parser.add_argument('--fi_model', type=str, default=None)  # , default='c4')
    parser.add_argument('--config', type=str, default='test')
    parser.add_argument('--omit_vars', type=str, default=None)  # comma-separated string of variables to omit
    parser.add_argument('--nosave_cols', type=str, default="prediction_model")

    ### Newly added arguments
    parser.add_argument('--folder_name', type=str, default=None)
    parser.add_argument('--fit_model', type=bool, default=False)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--n_cores', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default=default_dir)
    parser.add_argument('--simulation_seed', type=int, default=0)

    # arguments for rmd output of results
    parser.add_argument('--create_rmd', action='store_true', default=False)
    parser.add_argument('--show_vars', type=int, default=None)

    args = parser.parse_args()

    if args.parallel:
        if args.n_cores is None:
            print(os.getenv("SLURM_CPUS_ON_NODE"))
            n_cores = int(os.getenv("SLURM_CPUS_ON_NODE"))
        else:
            n_cores = args.n_cores
        client = Client(n_workers=n_cores)

    ests, fi_ests, \
    X_dgp, X_params_dict, y_dgp, y_params_dict, \
    vary_param_name, vary_param_vals = fi_config.get_fi_configs(args.config)

    metrics = get_metrics()

    if args.model:
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))
    if args.fi_model:
        fi_ests = list(filter(lambda x: args.fi_model.lower() == x[0].name.lower(), fi_ests))

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'sim', args.config, 'models', args.model, 'fi', args.fi_model)
    if len(fi_ests) == 0:
        raise ValueError('No valid FI estimators', 'sim', args.config, 'models', args.model, 'fi', args.fi_model)
    if args.verbose:
        print('running', args.config,
              'ests', ests,
              'fi_ests', fi_ests)
        print('\tsaving to', args.results_path)

    if args.omit_vars is not None:
        #results_dir = oj(args.results_path, args.config + "_omitted_vars")
        results_dir = oj(args.results_path, args.config + "_omitted_vars", args.folder_name)
    else:
        #results_dir = oj(args.results_path, args.config)
        results_dir = oj(args.results_path, args.config, args.folder_name)

    if isinstance(vary_param_name, list):
        #path = oj(results_dir, "varying_" + "_".join(vary_param_name), "seed" + str(args.split_seed))
        path = oj(results_dir, "varying_" + "_".join(vary_param_name), "seed" + str(args.simulation_seed))
    else:
        #path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.split_seed))
        path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.simulation_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    vary_type = None
    if isinstance(vary_param_name, list):  # multiple parameters are being varied
        # get parameters that are being varied over and identify whether it's a DGP/method/fi_method argument
        keys, values = zip(*vary_param_vals.items())
        vary_param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        vary_type = {}
        for vary_param_dict in vary_param_dicts:
            for param_name, param_val in vary_param_dict.items():
                if param_name in X_params_dict.keys() and param_name in y_params_dict.keys():
                    raise ValueError('Cannot vary over parameter in both X and y DGPs.')
                elif param_name in X_params_dict.keys():
                    vary_type[param_name] = "dgp"
                    X_params_dict[param_name] = vary_param_vals[param_name][param_val]
                elif param_name in y_params_dict.keys():
                    vary_type[param_name] = "dgp"
                    y_params_dict[param_name] = vary_param_vals[param_name][param_val]
                else:
                    est_kwargs = list(
                        itertools.chain(*[list(est.kwargs.keys()) for est in list(itertools.chain(*ests))]))
                    fi_est_kwargs = list(
                        itertools.chain(*[list(fi_est.kwargs.keys()) for fi_est in list(itertools.chain(*fi_ests))]))
                    if param_name in est_kwargs:
                        vary_type[param_name] = "est"
                    elif param_name in fi_est_kwargs:
                        vary_type[param_name] = "fi_est"
                    else:
                        raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp,
                                                 y_params_dict, y_dgp, ests, fi_ests, metrics, args) for i in
                    range(args.nreps)]
                results = dask.compute(*futures)
            else:
                # results = [
                #     run_simulation(i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp, y_params_dict,
                #                    y_dgp, ests, fi_ests, metrics, args) for i in range(args.nreps)]
                results = [
                    run_simulation(args.simulation_seed, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp, y_params_dict,
                                   y_dgp, ests, fi_ests, metrics, args) for i in range(args.nreps)]
            assert all(results)

    else:  # only on parameter is being varied over
        # get parameter that is being varied over and identify whether it's a DGP/method/fi_method argument
        for val_name, val in vary_param_vals.items():
            if vary_param_name in X_params_dict.keys() and vary_param_name in y_params_dict.keys():
                raise ValueError('Cannot vary over parameter in both X and y DGPs.')
            elif vary_param_name in X_params_dict.keys():
                vary_type = "dgp"
                X_params_dict[vary_param_name] = val
            elif vary_param_name in y_params_dict.keys():
                vary_type = "dgp"
                y_params_dict[vary_param_name] = val
            else:
                est_kwargs = list(itertools.chain(*[list(est.kwargs.keys()) for est in list(itertools.chain(*ests))]))
                fi_est_kwargs = list(
                    itertools.chain(*[list(fi_est.kwargs.keys()) for fi_est in list(itertools.chain(*fi_ests))]))
                if vary_param_name in est_kwargs:
                    vary_type = "est"
                elif vary_param_name in fi_est_kwargs:
                    vary_type = "fi_est"
                else:
                    raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests,
                                                 fi_ests, metrics, args) for i in range(args.nreps)]
                results = dask.compute(*futures)
            else:
                results = [
                    run_simulation(args.simulation_seed, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp, y_params_dict,
                                   y_dgp, ests, fi_ests, metrics, args) for i in range(args.nreps)]
                # results = [run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, fi_ests,
                #                           metrics, args) for i in range(args.nreps)]
            assert all(results)

    print('completed all experiments successfully!')

    # get model file names
    model_comparison_files_all = []
    for est in ests:
        estimator_name = est[0].name.split(' - ')[0]
        fi_estimators_all = [fi_estimator for fi_estimator in itertools.chain(*fi_ests) \
                             if fi_estimator.model_type in est[0].model_type]
        model_comparison_files = [f'{estimator_name}_{fi_estimator.name}_comparisons.pkl' for fi_estimator in
                                  fi_estimators_all]
        model_comparison_files_all += model_comparison_files

    # aggregate results
    results_list = []
    if isinstance(vary_param_name, list):
        for vary_param_dict in vary_param_dicts:
            val_name = "_".join(vary_param_dict.values())

            for i in range(args.nreps):
                all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                model_files = sorted([f for f in all_files if os.path.basename(f) in model_comparison_files_all])

                if len(model_files) == 0:
                    print('No files found at ', oj(path, val_name, 'rep' + str(i)))
                    continue

                results = pd.concat(
                    [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                    axis=0
                )

                for param_name, param_val in vary_param_dict.items():
                    val = vary_param_vals[param_name][param_val]
                    if vary_type[param_name] == "dgp":
                        if np.isscalar(val):
                            results.insert(0, param_name, val)
                        else:
                            results.insert(0, param_name, [val for i in range(results.shape[0])])
                        results.insert(1, param_name + "_name", param_val)
                    elif vary_type[param_name] == "est" or vary_type[param_name] == "fi_est":
                        results.insert(0, param_name + "_name", copy.deepcopy(results[param_name]))
                results.insert(0, 'rep', i)
                results_list.append(results)
    else:
        for val_name, val in vary_param_vals.items():
            for i in range(args.nreps):
                all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                model_files = sorted([f for f in all_files if os.path.basename(f) in model_comparison_files_all])

                if len(model_files) == 0:
                    print('No files found at ', oj(path, val_name, 'rep' + str(i)))
                    continue

                results = pd.concat(
                    [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                    axis=0
                )
                if vary_type == "dgp":
                    if np.isscalar(val):
                        results.insert(0, vary_param_name, val)
                    else:
                        results.insert(0, vary_param_name, [val for i in range(results.shape[0])])
                    results.insert(1, vary_param_name + "_name", val_name)
                    results.insert(2, 'rep', i)
                elif vary_type == "est" or vary_type == "fi_est":
                    results.insert(0, vary_param_name + "_name", copy.deepcopy(results[vary_param_name]))
                    results.insert(1, 'rep', i)
                results_list.append(results)
    results_merged = pd.concat(results_list, axis=0)
    pkl.dump(results_merged, open(oj(path, 'results.pkl'), 'wb'))
    results_df = reformat_results(results_merged)
    results_df.to_csv(oj(path, 'results.csv'), index=False)

    print('merged and saved all experiment results successfully!')

    # create R markdown summary of results
    if args.create_rmd:
        if args.show_vars is None:
            show_vars = 'NULL'
        else:
            show_vars = args.show_vars

        if isinstance(vary_param_name, list):
            vary_param_name = "; ".join(vary_param_name)

        sim_rmd = os.path.basename(results_dir) + '_simulation_results.Rmd'
        os.system(
            'cp {} \'{}\''.format(oj("rmd", "simulation_results.Rmd"), sim_rmd)
        )
        os.system(
            'Rscript -e "rmarkdown::render(\'{}\', params = list(results_dir = \'{}\', vary_param_name = \'{}\', seed = {}, keep_vars = {}), output_file = \'{}\', quiet = TRUE)"'.format(
                sim_rmd,
                results_dir, vary_param_name, str(args.split_seed), str(show_vars),
                oj(path, "simulation_results.html"))
        )
        os.system('rm \'{}\''.format(sim_rmd))
        print("created rmd of simulation results successfully!")