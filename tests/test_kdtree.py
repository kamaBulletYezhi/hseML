from sklearn.neighbors import KDTree as rightKD
#from sklearn.model_selection import train_test_split
from src.features.kdtree import KDTree
from src.data.splitdataset import train_test_split
import numpy as np
import pytest


n = 100
d = 5
t = 10


@pytest.mark.parametrize("X", [np.random.randn(n, d) for _ in range(t)])
@pytest.mark.parametrize("y", [np.random.randn(n, d) for _ in range(t)])
@pytest.mark.parametrize("leaf_size", [np.random.randint(5, 20) for _ in range(3)])
def test_kd_tree(X, y, leaf_size):
    n_neighbors = 4
    my_tree = KDTree(X, leaf_size, d)
    true_tree = rightKD(X, leaf_size, 'euclidean')
    neg1 = my_tree.query(y, n_neighbors)
    _, neg2 = true_tree.query(y, n_neighbors)
    assert (neg1 == neg2).all()

