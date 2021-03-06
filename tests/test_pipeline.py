import os
import pickle as pkl
import sys
from os.path import dirname
from os.path import join

import pytest
import sh

repo_dir = dirname(dirname(os.path.abspath(__file__)))
print('repo_dir', repo_dir)
sys.path.append(repo_dir)
import config
import shutil


def test_valid_configs():
    print('TESTSTSE\n\n', config.X)
    config_dir = join(repo_dir, 'config')
    configs = [d for d in os.listdir(config_dir)
               if os.path.isdir(join(config_dir, d))
               and not d.startswith('_')
               and not d in ['stablerules', 'interactions']  # omit configs under active dev
               ]
    for c in configs:
        DSETS_CLASSIFICATION, DSETS_REGRESSION, ESTS_CLASSIFICATION, ESTS_REGRESSION = config.get_configs(c)


def test_fit_models():
    test_dir = join(repo_dir, 'results_test')
    print(repo_dir)
    try:
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        sh.python(['01_fit_models.py', '--dataset', 'heart',
                   '--model', 'cart', '--config', 'test',
                   '--results_path', test_dir,
                   '--split_seed', '0',
                   '--ignore_cache'])
        assert os.path.isdir(join(test_dir, 'test', 'heart', 'train-test'))
        assert os.path.isfile(join(test_dir, 'test', 'heart', 'train-test', 'seed0', 'CART_comparisons.pkl'))
        sh.python(['01_fit_models.py', '--dataset', 'heart',
                   '--model', 'cart', '--config', 'test',
                   '--results_path', test_dir,
                   '--split_seed', '1',
                   '--ignore_cache'])

        sh.python(['02_aggregate_results.py',
                   '--results_path', test_dir])
        assert os.path.isfile(join(test_dir, 'test', 'heart', 'train-test', 'seed0', 'results_aggregated.pkl'))
        results = pkl.load(open(join(test_dir, 'test', 'heart', 'train-test', 'seed0', 'results_aggregated.pkl'), 'rb'))
        assert 'df' in results

        # mean stuff
        assert os.path.isfile(join(test_dir, 'test', 'heart', 'train-test', 'results_aggregated.pkl'))
        results = pkl.load(open(join(test_dir, 'test', 'heart', 'train-test', 'results_aggregated.pkl'), 'rb'))
        assert 'df' in results
        assert 'df_mean' in results
        assert 'df_rules' in results

    except sh.ErrorReturnCode as e:
        print(e)
        pytest.fail(e)
    finally:
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
