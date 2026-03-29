"""Evaluation helpers for employer-facing model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

from .config import EVALUATION_THRESHOLDS, METRICS_JSON_FILE, RESULTS_DIR, SUMMARY_TABLE_FILE


def ensure_results_directory() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_training_curves(history, method_name: str) -> None:
    """Save training curves so reviewers can see model stability and convergence."""
    ensure_results_directory()
    metrics_to_plot = ["loss", "pr_auc", "precision", "recall"]
    plt.figure(figsize=(12, 8))

    for plot_index, metric_name in enumerate(metrics_to_plot, start=1):
        plt.subplot(2, 2, plot_index)
        plt.plot(history.history[metric_name], label="Train")
        plt.plot(history.history[f"val_{metric_name}"], label="Validation")
        plt.title(f"{method_name} - {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{method_name}_training_curves.png")
    plt.close()


def save_confusion_matrix(labels, predictions, threshold: float, method_name: str) -> dict:
    """Save confusion matrices using injury-claim language instead of generic fraud terminology."""
    ensure_results_directory()
    binary_predictions = (predictions >= threshold).astype(int)
    cm = confusion_matrix(labels, binary_predictions)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(f"{method_name} Confusion Matrix @ {threshold:.2f}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{method_name}_confusion_matrix_threshold_{str(threshold).replace('.', '_')}.png")
    plt.close()

    return {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }


def evaluate_method(method_name: str, model, test_features, test_labels, thresholds=None) -> list[dict]:
    """Evaluate a model across one or more thresholds and return employer-friendly metrics."""
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS

    predicted_probabilities = model.predict(test_features, verbose=0).flatten()
    pr_auc = average_precision_score(test_labels, predicted_probabilities)

    rows = []
    for threshold in thresholds:
        binary_predictions = (predicted_probabilities >= threshold).astype(int)
        confusion_summary = save_confusion_matrix(
            test_labels,
            predicted_probabilities,
            threshold,
            method_name,
        )

        rows.append(
            {
                "method": method_name,
                "threshold": threshold,
                "precision": float(precision_score(test_labels, binary_predictions, zero_division=0)),
                "recall": float(recall_score(test_labels, binary_predictions, zero_division=0)),
                "pr_auc": float(pr_auc),
                **confusion_summary,
            }
        )

    return rows


def save_method_summary(summary_rows: list[dict]) -> pd.DataFrame:
    """Create and save a final comparison table across all methods."""
    ensure_results_directory()
    summary_df = pd.DataFrame(summary_rows).sort_values(["pr_auc", "recall"], ascending=False)
    summary_df.to_csv(SUMMARY_TABLE_FILE, index=False)

    with open(METRICS_JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)

    return summary_df
