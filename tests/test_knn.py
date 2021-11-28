from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
from src.features.knn import KNearest
from src.data.splitdataset import train_test_split
import numpy as np
import pytest

n = 100
i = 10


@pytest.mark.parametrize("X", [np.random.randn(n, 2) for _ in range(i)])
@pytest.mark.parametrize("y", [np.random.randint(0, 5, n) for _ in range(i)])
@pytest.mark.parametrize("n_neighbors", [np.random.randint(2, 6) for _ in range(i)])
@pytest.mark.parametrize("ratio", [0.1 + 0.4 * np.random.random_sample()])
def test_knn(X, y, n_neighbors, ratio):
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=ratio)

    true_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    true_knn.fit(X_train, y_train)
    true_pred = true_knn.predict(X_test)

    my_knn = KNearest(n_neighbors)
    my_knn.fit(X_train, y_train)
    my_pred = my_knn.predict(X_test)

    assert (my_pred == true_pred).all()
