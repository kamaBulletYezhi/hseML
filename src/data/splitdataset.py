import numpy as np
from typing import Tuple


def train_test_split(X: np.array, y: np.array, ratio: float
                     ) -> Tuple[np.array, np.array, np.array, np.array]:
    n = int(len(X) * ratio)
    X_train = X[:n]
    y_train = y[:n]
    X_test = X[n:]
    y_test = y[n:]
    return X_train, y_train, X_test, y_test
