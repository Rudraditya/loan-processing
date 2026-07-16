"""Evaluation metrics and diagnostic plots for the risk models."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No Default", "Default"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """`results` maps model name -> (y_true, y_proba)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, (y_true, y_proba) in results.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model, feature_names: list[str], output_path: Path, top_n: int = 15
) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#2b6cb0")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
