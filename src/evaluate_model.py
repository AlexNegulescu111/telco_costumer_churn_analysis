import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict


def oof_probabilities(pipeline, x_train, y_train, cv, fit_params=None):
    """Out-of-fold churn probabilities on the training set (each row predicted by a model that never saw it)."""
    return cross_val_predict(pipeline, x_train, y_train, cv=cv, method="predict_proba", params=fit_params)[:, 1]


def find_best_threshold(y_true, proba):
    """Decision threshold that maximises F1, chosen on out-of-fold training predictions (never on test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the last point has no threshold
    return float(thresholds[best])


def compute_metrics(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def metrics_table(results):
    """results: {model_name: metrics dict} -> DataFrame without the confusion matrices."""
    rows = {name: {k: v for k, v in m.items() if k != "confusion_matrix"} for name, m in results.items()}
    return pd.DataFrame(rows).T.round(4)


def plot_evaluation(y_true, probas, thresholds, path):
    """ROC curves, precision-recall curves and confusion matrices of all models on the test set."""
    names = list(probas)
    fig, axes = plt.subplots(2, max(len(names), 2), figsize=(5 * max(len(names), 2), 9))

    for name in names:
        RocCurveDisplay.from_predictions(y_true, probas[name], name=name, ax=axes[0, 0])
        PrecisionRecallDisplay.from_predictions(y_true, probas[name], name=name, ax=axes[0, 1])
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set_title("ROC curve (test set)")
    axes[0, 1].axhline(np.mean(y_true), color="k", linestyle="--", linewidth=1, label="Baseline (churn rate)")
    axes[0, 1].set_title("Precision-recall curve (test set)")
    axes[0, 1].legend(loc="lower left")
    for ax in axes[0, 2:]:
        ax.axis("off")

    for ax, name in zip(axes[1], names):
        y_pred = (probas[name] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Stay", "Churn"], ax=ax, colorbar=False)
        ax.set_title(f"{name} (threshold {thresholds[name]:.2f})")
    for ax in axes[1, len(names) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
