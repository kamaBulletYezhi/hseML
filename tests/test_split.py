import numpy as np
from src.data.splitdataset import train_test_split
import pytest


@pytest.mark.parametrize("X", [np.random.uniform(-1000, 1000, (50, 5)) for _ in range(5)])
@pytest.mark.parametrize("y", [np.random.randint(0, 10, 50) for _ in range(5)])
@pytest.mark.parametrize("ratio", [np.random.uniform(0.1, 0.9) for _ in range(5)])
def test_shape(X, y, ratio):
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert int(X.shape[0] * ratio) == X_train.shape[0]
    assert int(y.shape[0] * ratio) == y_train.shape[0]