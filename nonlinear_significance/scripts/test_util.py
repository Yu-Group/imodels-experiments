import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.append("..")
from scripts.util import *


def test_compare():
    query = [0, 1, 2]
    threshold = 1.5
    assert not compare(query, 0, threshold)
    assert not compare(query, 1, threshold)
    assert compare(query, 2, threshold)
    assert compare(query, 1, threshold, False)


def test_make_stumps():
    n = 200
    d = 2
    sigma = 0

    X = np.random.randn(n, d)

    def spike(x):
        return all(x > 0)

    y = np.apply_along_axis(spike, 1, X) + np.random.randn(n) * sigma
    decision_tree = DecisionTreeRegressor(max_depth=2)
    decision_tree.fit(X, y)
    tree_struct = decision_tree.tree_
    stumps = make_stumps(decision_tree.tree_)
    assert len(stumps) == 2
    assert len(stumps[1].a_features) == 1

    # Test normalization
    stumps = make_stumps(decision_tree.tree_, normalize=True)
    assert np.isclose(np.apply_along_axis(stumps[0], 1, X).sum(), 0)
    assert np.isclose(np.sum(np.apply_along_axis(stumps[0], 1, X) ** 2), n)


def test_local_decision_stump():
    stump = LocalDecisionStump(feature=0, threshold=0, left_val=-4,
                               right_val=5, a_features=[], a_thresholds=[], a_signs=[])
    assert stump([1, 0]) == 5
    assert stump([-1, 0]) == -4

    stump = LocalDecisionStump(feature=1, threshold=0, left_val=-0.5,
                               right_val=0.5, a_features=[0], a_thresholds=[2], a_signs=[True])
    assert stump([1, 0]) == 0
    assert stump([3, 2]) == 0.5
