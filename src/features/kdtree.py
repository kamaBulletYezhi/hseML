import numpy as np
from typing import List
from heapq import nsmallest


class KDNode:
    def __init__(self, ind: int, med: float, parent = None, dots = None):
        self.its_leaf = False
        self.parent = parent
        self.dots = dots
        if dots is None:
            self.index = ind
            self.med = med
            self.left = None
            self.right = None
        else:
            self.its_leaf = True


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40, N: int = 10):
        # N --- сколько фичей надо побробовать перед тем, как записать точки в один лист
        # по хорошему N = X.shape[1], но так очень долго считается
        def meeting_parents(node, parent, left_child):
            if parent is not None:
                if left_child:
                    parent.left = node
                else:
                    parent.right = node

        def builder(i: int, indices: np.array, parent: KDNode, left_child: bool) -> KDNode:
            if sum(indices) < 2 * leaf_size:
                node = KDNode(0, 0, parent, indices)
                meeting_parents(node, parent, left_child)
                return node

            # поиск валидной фичи
            for _ in range(N):
                med = np.median(self.X[indices, i], axis=0)
                left_indices = (self.X[:, i] < med) & indices
                right_indices = (self.X[:, i] >= med) & indices
                if min(sum(left_indices), sum(right_indices)) >= leaf_size:
                    break
                i = (i + 1) % N
            else:
                node = KDNode(0, 0, parent, indices)
                meeting_parents(node, parent, left_child)
                return node

            # запись узла и присоединение к дереву
            node = KDNode(i, med, parent, indices)
            meeting_parents(node, parent, left_child)
            i = (i + 1) % N

            node.left = builder(i, left_indices, node, True)
            node.right = builder(i, right_indices, node, False)
            return node

        self.X = X
        m = self.X.shape[0]
        self.root = builder(0, np.array([True]*m), None, False)

    def query(self, X: np.array, k: int = 1) -> List[List]:
        # Евклидово расстояние между point и X[ind]
        def dist_to(point):
            def inner(ind):  # I don't like anonymous functions
                return np.linalg.norm(self.X[ind] - point)
            return inner

        def merge(arr1: list, arr2: list, x) -> List:
            arr = []
            j_1 = 0
            j_2 = 0
            while j_1 < len(arr1) and j_2 < len(arr2):
                if dist_to(x)(arr1[j_1]) < dist_to(x)(arr2[j_2]):
                    arr.append(arr1[j_1])
                    j_1 += 1
                else:
                    arr.append(arr2[j_2])
                    j_2 += 1
            arr += arr1[j_1:] + arr2[j_2:]
            return arr[: k]

        def get_neighbors(x: np.array, node: KDNode) -> List:
            def sorted_dots(node):
                return nsmallest(k, np.where(node.dots)[0], key=dist_to(x))
            if node.its_leaf:
                return sorted_dots(node)

            ind, med = node.index, node.med
            first_path, second_path = (node.left, node.right) if x[ind] < med else (node.right, node.left)
            first_neighbors = get_neighbors(x, first_path)
            max_dist = first_neighbors[-1]
            if len(first_neighbors) == k and max_dist < abs(x[ind] - med):
                return first_neighbors
            second_neighbors = get_neighbors(x, second_path)
            return merge(first_neighbors, second_neighbors, x)

        neighbors = [[] for _ in X]
        for i, x in enumerate(X):
            neighbors[i] = get_neighbors(x, self.root)

        return neighbors

