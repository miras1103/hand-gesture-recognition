from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.experiments import calculate_metrics, save_experiment_plots, write_metrics_csv
from src.model import build_classifier
from src.preprocessing import CLASSES, load_dataset

DEFAULT_DATA_DIR = WORKSPACE_DIR / "dataset"
RESULTS_DIR = PROJECT_DIR / "results"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
SEED = 42


def create_split(
    images: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
) -> tuple[np.ndarray, ...]:
    """Match the notebook's stratified 80/20 test then 80/20 train/validation split."""
    x_train_val, x_test, y_train_val, y_test, paths_train_val, paths_test = (
        train_test_split(
            images,
            labels,
            paths,
            test_size=0.2,
            random_state=SEED,
            stratify=labels,
        )
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test, paths_test


def train_experiment(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    use_augmentation: bool,
    epochs: int,
    weights: str | None,
) -> dict[str, str | int | float]:
    x_train, x_val, x_test, y_train, y_val, y_test, paths_test = create_split(
        images, labels, paths
    )
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)

    model = build_classifier(augmentation=use_augmentation, weights=weights)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    experiment_name = (
        "efficientnetv2b0_with_augmentation"
        if use_augmentation
        else "efficientnetv2b0_without_augmentation"
    )
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    metrics = calculate_metrics(
        y_test,
        y_pred,
        model_name=experiment_name,
        augmentation=use_augmentation,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS_DIR / f"{experiment_name}.keras")
    save_experiment_plots(
        y_test,
        y_pred,
        history.history,
        RESULTS_DIR / "figures",
        experiment_name,
    )

    history_path = RESULTS_DIR / f"{experiment_name}_history.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_keys = list(history.history)
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *history_keys])
        writer.writeheader()
        for epoch_index in range(len(history.history["loss"])):
            writer.writerow(
                {
                    "epoch": epoch_index + 1,
                    **{
                        key: values[epoch_index]
                        for key, values in history.history.items()
                    },
                }
            )

    predictions_path = RESULTS_DIR / f"{experiment_name}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class", "predicted_class", *CLASSES])
        for image_path, true_label, predicted_label, scores in zip(
            paths_test, y_test, y_pred, probabilities, strict=True
        ):
            writer.writerow(
                [
                    image_path,
                    CLASSES[int(true_label)],
                    CLASSES[int(predicted_label)],
                    *[float(score) for score in scores],
                ]
            )

    split_metadata = {
        "seed": SEED,
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "class_order": CLASSES,
        "image_size": [128, 128],
        "weights": weights or "random_initialization",
    }
    (RESULTS_DIR / f"{experiment_name}_metadata.json").write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )
    return metrics


def write_comparison(rows: list[dict[str, str | int | float]]) -> Path:
    baseline_path = PROJECT_DIR / "results" / "baseline_reference.csv"
    with baseline_path.open(newline="", encoding="utf-8") as file:
        baseline_rows = list(csv.DictReader(file))

    comparison_path = RESULTS_DIR / "comparison.csv"
    fields = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "source",
        "status",
    ]
    measured_rows = [
        {
            "model": row["model"],
            "accuracy": row["accuracy"],
            "precision_weighted": row["precision_weighted"],
            "recall_weighted": row["recall_weighted"],
            "f1_weighted": row["f1_weighted"],
            "source": "Measured by Assignment 2 training script on held-out test split",
            "status": "measured",
        }
        for row in rows
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(baseline_rows + measured_rows)
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2B0 with a controlled augmentation ablation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset folder containing class subfolders (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--experiment",
        choices=("both", "with_augmentation", "without_augmentation"),
        default="both",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="ImageNet weights require a one-time download; 'none' initializes randomly.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    images, labels, paths = load_dataset(args.data_dir)
    print(f"Loaded {len(images)} images from {args.data_dir.resolve()}")
    for class_index, class_name in enumerate(CLASSES):
        print(f"  {class_name}: {int(np.sum(labels == class_index))}")

    experiments = {
        "with_augmentation": True,
        "without_augmentation": False,
    }
    selected = (
        list(experiments)
        if args.experiment == "both"
        else [args.experiment]
    )
    rows = []
    for name in selected:
        print(f"\nTraining EfficientNetV2B0: {name}")
        rows.append(
            train_experiment(
                images=images,
                labels=labels,
                paths=paths,
                use_augmentation=experiments[name],
                epochs=args.epochs,
                weights="imagenet" if args.weights == "imagenet" else None,
            )
        )

    write_metrics_csv(rows, RESULTS_DIR / "assignment2_metrics.csv")
    comparison_path = write_comparison(rows)
    print(f"\nMeasured test metrics saved to {RESULTS_DIR / 'assignment2_metrics.csv'}")
    print(f"Baseline and measured Assignment 2 comparison saved to {comparison_path}")
    print("No result is reported for an experiment that was not run.")


if __name__ == "__main__":
    main()
