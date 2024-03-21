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
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, mean_squared_error
from sklearn import preprocessing

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import fi_config
from util import ModelConfig, FIModelConfig, tp, fp, neg, pos, specificity_score, auroc_score, auprc_score, compute_nsg_feat_corr_w_sig_subspace, apply_splitting_strategy

warnings.filterwarnings("ignore", message="Bins whose width")

#RUN THE FILE
# python 01_run_ablation_classification.py --nreps 5 --config mdi_local.real_data_classification --split_seed 331 --ignore_cache --create_rmd --result_name diabetes_classification


def generate_random_shuffle(data, seed):
    """
    Randomly shuffle each column of the data.
    """
    np.random.seed(seed)
    return np.array([np.random.permutation(data[:, i]) for i in range(data.shape[1])]).T


def ablation(data, feature_importance, mode, num_features, seed):
    """
    Replace the top num_features max feature importance data with random shuffle for each sample
    """
    assert mode in ["max", "min"]
    fi = feature_importance.to_numpy()
    shuffle = generate_random_shuffle(data, seed)
    if mode == "max":
        indices = np.argsort(-fi)
    else:
        indices = np.argsort(fi)
    data_copy = data.copy()
    for i in range(data.shape[0]):
        for j in range(num_features):
            data_copy[i, indices[i,j]] = shuffle[i, indices[i,j]]
    return data_copy


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
    feature_importance_list = []

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
                X_tune = X
                X_test = X
                y_train = y
                y_tune = y
                y_test = y

            normalizer = preprocessing.Normalizer()
            if args.normalization == "train_test":
                X_train = normalizer.fit_transform(X_train)
                X_test = normalizer.transform(X_test)
            elif args.normalization == "all":
                X = normalizer.fit_transform(X)
                X_train = normalizer.transform(X_train)
                X_test = normalizer.transform(X_test)

            # fit model
            est.fit(X_train, y_train)
            test_all_auc = roc_auc_score(y_test, est.predict_proba(X_test)[:, 1])
            test_all_auprc = auprc_score(y_test, est.predict_proba(X_test)[:, 1])
            test_all_f1 = f1_score(y_test, est.predict_proba(X_test)[:, 1] > 0.5)

            if model.name == "RF_plus":
                indices = np.random.choice(X_test.shape[0], 100, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]

            # loop over fi estimators
            rng = np.random.RandomState()
            number_of_ablations = 10
            seeds = rng.randint(0, 10000, number_of_ablations)
            for fi_est in tqdm(fi_ests):
                metric_results = {
                    'model': model.name,
                    'fi': fi_est.name,
                    'splitting_strategy': splitting_strategy,
                    'train_size': X_train.shape[0],
                    'num_features': X_train.shape[1],
                    'data_split_seed': args.split_seed,
                    'test_size': X_test.shape[0],
                    'test_all_auc': test_all_auc,
                    'test_all_auprc': test_all_auprc,
                    'test_all_f1': test_all_f1
                }
                for i in range(100):
                    if model.name == "RF_plus":
                        metric_results[f'sample_test_{i}'] = indices[i]
                    else:
                        metric_results[f'sample_test_{i}'] = None
                for i in range(len(seeds)):
                    metric_results[f'ablation_seed_{i}'] = seeds[i]
                start = time.time()
                local_fi_score = fi_est.cls(X_train=X_train, y_train=y_train, 
                                            X_test=X_test, y_test=y_test, 
                                            fit =copy.deepcopy(est), **fi_est.kwargs)
                end = time.time()
                metric_results['fi_time'] = end - start
                feature_importance_list.append(local_fi_score)
                support_df = pd.DataFrame({"var": np.arange(len(support)),
                                           "true_support": support})#,
                                           #"cor_with_signal": x_cor})
                metric_results['fi_scores'] = support_df

                start = time.time()
                y_pred = est.predict_proba(X_test)[:, 1]
                metric_results['AUROC_before_ablation'] = roc_auc_score(y_test, y_pred)
                metric_results['AUPRC_before_ablation'] = auprc_score(y_test, y_pred)
                metric_results['F1_before_ablation'] = f1_score(y_test, y_pred > 0.5)
                imp_vals = copy.deepcopy(local_fi_score)
                imp_vals[imp_vals == float("-inf")] = -sys.maxsize - 1
                imp_vals[imp_vals == float("inf")] = sys.maxsize - 1
                ablation_results_auroc_list = [0] * X_test.shape[1]
                ablation_results__auprc_list = [0] * X_test.shape[1]
                ablation_results_f1_list = [0] * X_test.shape[1]
                for seed in tqdm(seeds):
                    for i in range(X_test.shape[1]):
                        if fi_est.ascending:
                            ablation_X_test = ablation(X_test, imp_vals, "max", i+1, seed)
                        else:
                            ablation_X_test = ablation(X_test, imp_vals, "min", i+1, seed)
                        ablation_results_auroc_list[i] += roc_auc_score(y_test, est.predict_proba(ablation_X_test)[:, 1])
                        ablation_results__auprc_list[i] += auprc_score(y_test, est.predict_proba(ablation_X_test)[:, 1])
                        ablation_results_f1_list[i] += f1_score(y_test, est.predict_proba(ablation_X_test)[:, 1] > 0.5)
                ablation_results_f1_list = [x / number_of_ablations for x in ablation_results_f1_list]
                ablation_results_auroc_list = [x / number_of_ablations for x in ablation_results_auroc_list]
                ablation_results__auprc_list = [x / number_of_ablations for x in ablation_results__auprc_list]
                for i in range(X_test.shape[1]):
                    metric_results[f'AUROC_after_ablation_{i+1}'] = ablation_results_auroc_list[i]
                    metric_results[f'AUPRC_after_ablation_{i+1}'] = ablation_results__auprc_list[i]
                    metric_results[f'F1_after_ablation_{i+1}'] = ablation_results_f1_list[i]
                end = time.time()
                metric_results['ablation_time'] = end - start
                print(f"data_size: {X_test.shape[0]}, fi: {fi_est.name}, done with time: {end - start}")

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
    
    feature_importance_all = [oj(path, f'{estimator_name}_{fi_estimator.name}_feature_importance.pkl') \
                                  for fi_estimator in fi_estimators_all]


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

    for i in range(len(feature_importance_all)):
        pkl.dump(fi_lst[i], open(feature_importance_all[i], 'wb'))

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
    fi_scores = pd.concat(results.pop('fi_scores').to_dict()). \
        reset_index(level=0).rename(columns={'level_0': 'index'})
    results_df = pd.merge(results, fi_scores, left_index=True, right_on="index")
    return results_df


def run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, fi_ests, metrics, args):
    os.makedirs(oj(path, val_name, "rep" + str(i)), exist_ok=True)
    np.random.seed(i)
    max_iter = 100
    iter = 0
    while iter <= max_iter:  # regenerate data if y is constant
        X = X_dgp(**X_params_dict)
        y, support, beta = y_dgp(X, **y_params_dict, return_support=True)
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
    parser.add_argument('--result_name', type=str, default=None)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--n_cores', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default=default_dir)
    parser.add_argument('--normalization', type=str, default="none")

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
        results_dir = oj(args.results_path, args.config + "_omitted_vars", args.result_name)
    else:
        #results_dir = oj(args.results_path, args.config)
        results_dir = oj(args.results_path, args.config, args.result_name)

    if isinstance(vary_param_name, list):
        path = oj(results_dir, "varying_" + "_".join(vary_param_name), "seed" + str(args.split_seed))
    else:
        path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.split_seed))
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
                results = [
                    run_simulation(i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp, y_params_dict,
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
                results = [run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, fi_ests,
                                          metrics, args) for i in range(args.nreps)]
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