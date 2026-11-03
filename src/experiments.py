from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing import CLASSES

METRIC_FIELDS = (
    "model",
    "augmentation",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "test_samples",
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    augmentation: bool,
) -> dict[str, str | int | float]:
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "model": model_name,
        "augmentation": augmentation,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "test_samples": int(len(y_true)),
    }


def write_metrics_csv(
    rows: list[dict[str, str | int | float]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_experiment_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: dict[str, list[float]],
    output_dir: str | Path,
    experiment_name: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if history:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric, label in (
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "loss", "Loss"),
        ):
            axis.plot(history.get(metric, []), label="Train")
            axis.plot(history.get(f"val_{metric}", []), label="Validation")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.set_title(f"{label} by epoch")
            axis.legend()
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / f"{experiment_name}_training_curves.png", dpi=160)
        plt.close(figure)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    figure, axis = plt.subplots(figsize=(5, 4))
    display.plot(ax=axis, cmap="Purples", colorbar=False, values_format="d")
    axis.set_title(f"{experiment_name} — test confusion matrix")
    figure.tight_layout()
    figure.savefig(output / f"{experiment_name}_confusion_matrix.png", dpi=160)
    plt.close(figure)
