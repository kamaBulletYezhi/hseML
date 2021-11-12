import numpy as np
from typing import NoReturn, List
from src.features.kdtree import KDTree


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.X = None
        self.y = None
        self.labels = None
        self.tree = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        self.X = X
        self.y = y
        self.labels = sorted(np.unique(y))
        self.tree = KDTree(self.X, self.leaf_size)

    def predict_proba(self, X: np.array) -> List[np.array]:
        predict_k_labels = self.y[self.tree.query(X, k=self.n_neighbors)]

        probability = np.zeros((X.shape[0], len(self.labels)))
        for i, l in enumerate(self.labels):
            probability[:, i] += (predict_k_labels == l).sum(1)
        probability /= self.n_neighbors
        return probability

    def predict(self, X: np.array) -> np.array:
        return np.argmax(self.predict_proba(X), axis=1)
