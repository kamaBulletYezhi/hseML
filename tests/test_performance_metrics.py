import numpy as np
import pytest
from src.reports.performance_metrics import get_precision_recall_accuracy as g_pra
from sklearn.metrics import precision_score, recall_score, accuracy_score


@pytest.mark.parametrize("y_true", [np.random.randint(0, 4, 50) for _ in range(20)])
@pytest.mark.parametrize("y_pred", [np.random.randint(0, 4, 50) for _ in range(20)])
class TestPerformanceMetrics:

    def test_precision(self, y_pred, y_true):
        assert (precision_score(y_true, y_pred, average=None) == g_pra(y_pred, y_true)[0]).all()

    def test_recall(self, y_pred, y_true):
        assert (recall_score(y_true, y_pred, average=None) == g_pra(y_pred, y_true)[1]).all()

    def test_accuracy(self, y_pred, y_true):
        assert accuracy_score(y_true, y_pred) == g_pra(y_pred, y_true)[2]
