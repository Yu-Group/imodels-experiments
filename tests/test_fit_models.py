import os
from os.path import dirname
from os.path import join

import pytest
import sh

repo_dir = dirname(dirname(os.path.abspath(__file__)))
import shutil


def test_install():
    # cwd = os.getcwd()
    # os.chdir(repo_dir)
    test_dir = join(repo_dir, 'results_test')
    print(repo_dir)
    try:
        # run the shell command
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        sh.python(['01_fit_models.py', '--dataset', 'heart',
                   '--model', 'cart', '--config', 'test',
                   '--results_path', test_dir])
        assert os.path.isdir(join(test_dir, 'test', 'heart', 'train-test'))
        assert os.path.isfile(join(test_dir, 'test', 'heart', 'train-test', 'seed0', 'CART_comparisons.pkl'))
    except sh.ErrorReturnCode as e:
        print(e)
        pytest.fail(e)
    finally:
        # run the shell command
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
