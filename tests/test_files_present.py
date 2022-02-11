import os
import sys
from os.path import dirname
from os.path import join

repo_dir = dirname(dirname(os.path.abspath(__file__)))
print('repo_dir', repo_dir)
sys.path.append(repo_dir)


def test_files_present():
    """Just makes sure certain files that are externally used don't get moved
    """
    diabetes_fig = join(repo_dir, 'docs/figs/diabetes_figs.svg')
    assert os.path.exists(diabetes_fig), 'diabetes fig should not be moved'
