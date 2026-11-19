import numpy as np
import pytest

from evaluate_model import compute_metrics, find_best_threshold, metrics_table


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    threshold = find_best_threshold(y, proba)
    assert 0.3 < threshold <= 0.6
    assert compute_metrics(y, proba, threshold)["f1"] == 1.0


def test_compute_metrics():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.2, 0.8, 0.6, 0.4])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["accuracy"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]


def test_metrics_table_drops_confusion_matrix():
    y = np.array([0, 1, 0, 1])
    table = metrics_table({"A": compute_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]))})
    assert "confusion_matrix" not in table.columns
    assert table.loc["A", "roc_auc"] == 1.0
