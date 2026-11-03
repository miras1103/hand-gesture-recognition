from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.train import SEED, create_split
from src.experiments import calculate_metrics, save_experiment_plots, write_metrics_csv
from src.model import build_classifier
from src.preprocessing import CLASSES, load_dataset, load_rgb_array

DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_ORIGINAL_DATA_DIR = WORKSPACE_DIR / "dataset"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "artaug"
RESULTS_DIR = PROJECT_DIR / "results" / "artaug"
EXPECTED_ARTAUG_PER_CLASS = 10
EXPECTED_ORIGINAL_SPLIT = (196, 49, 62)
BATCH_SIZE = 16
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
EXPERIMENT_NAME = "efficientnetv2b0_artaug"
CHECKPOINT_PATH = ARTIFACTS_DIR / f"{EXPERIMENT_NAME}.keras"
OUTPUT_FILES = (
    CHECKPOINT_PATH,
    RESULTS_DIR / "assignment2_metrics.csv",
    RESULTS_DIR / f"{EXPERIMENT_NAME}_history.csv",
    RESULTS_DIR / f"{EXPERIMENT_NAME}_metadata.json",
    RESULTS_DIR / f"{EXPERIMENT_NAME}_predictions.csv",
    RESULTS_DIR / f"{EXPERIMENT_NAME}_classification_report.csv",
    RESULTS_DIR / "baseline_vs_artaug.csv",
    RESULTS_DIR / "figures" / f"{EXPERIMENT_NAME}_training_curves.png",
    RESULTS_DIR / "figures" / f"{EXPERIMENT_NAME}_confusion_matrix.png",
)


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain these columns: {', '.join(sorted(required_columns))}"
            )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _load_approved_artaug(
    *,
    data_dir: Path,
    original_training_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    manifest_path = data_dir / "manifest.csv"
    review_path = data_dir / "artaug_review.csv"
    manifest_rows = _read_csv(
        manifest_path,
        {"filename", "class", "source", "original_filename", "split", "approval_status"},
    )
    review_rows = _read_csv(
        review_path,
        {
            "filename",
            "class",
            "technical_status",
            "manual_review_status",
            "accepted",
            "notes",
        },
    )

    reviews_by_filename: dict[str, dict[str, str]] = {}
    for row in review_rows:
        normalized = _normalize_relative_path(row["filename"])
        if normalized in reviews_by_filename:
            raise ValueError(f"Duplicate filename in review CSV: {row['filename']}")
        reviews_by_filename[normalized] = row

    original_training_path_set = {
        _normalize_relative_path(path) for path in original_training_paths
    }
    source_training_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in manifest_rows:
        if (
            row["source"].casefold() == "original"
            and row["split"].casefold() == "train"
            and row["approval_status"].casefold() == "approved"
            and row["class"] in CLASSES
        ):
            source_training_rows.setdefault(
                (row["class"], Path(row["original_filename"]).name.casefold()), []
            ).append(row)

    samples_by_class: dict[str, list[tuple[np.ndarray, str]]] = {
        class_name: [] for class_name in CLASSES
    }
    seen_output_paths: set[str] = set()
    seen_source_pairs: set[tuple[str, str]] = set()
    for row in manifest_rows:
        if (
            row["source"].casefold() != "artaug"
            or row["split"].casefold() != "train"
            or row["approval_status"].casefold() != "approved"
        ):
            continue
        class_name = row["class"]
        if class_name not in CLASSES:
            raise ValueError(f"Unexpected class for ArtAug sample: {class_name!r}")

        relative = PurePosixPath(row["filename"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 3
            or relative.parts[0] != "artaug_generated"
            or relative.parts[1] != class_name
        ):
            raise ValueError(f"Unsafe or misclassified ArtAug manifest path: {row['filename']}")
        normalized_path = _normalize_relative_path(row["filename"])
        if normalized_path in seen_output_paths:
            raise ValueError(f"Duplicate ArtAug manifest path: {row['filename']}")
        seen_output_paths.add(normalized_path)

        review = reviews_by_filename.get(normalized_path)
        if review is None:
            raise ValueError(f"Approved ArtAug image has no review row: {row['filename']}")
        if (
            review["class"] != class_name
            or review["technical_status"].casefold() != "passed"
            or review["manual_review_status"].casefold() != "approved"
            or review["accepted"].casefold() not in {"true", "yes", "1"}
        ):
            raise ValueError(f"ArtAug image is not fully approved in review CSV: {row['filename']}")

        generated_path = data_dir.joinpath(*relative.parts)
        if not generated_path.is_file():
            raise FileNotFoundError(f"Approved ArtAug image is missing: {generated_path}")
        if not generated_path.stem.casefold().endswith("_artaug"):
            raise ValueError(f"ArtAug output must use the _artaug suffix: {generated_path.name}")

        original_filename = Path(row["original_filename"]).name
        source_pair = (class_name, original_filename.casefold())
        if source_pair in seen_source_pairs:
            raise ValueError(
                f"Multiple ArtAug outputs map to the same original: {class_name}/{original_filename}"
            )
        seen_source_pairs.add(source_pair)
        source_rows = source_training_rows.get(source_pair, [])
        if len(source_rows) != 1:
            raise ValueError(
                f"Expected exactly one approved original train manifest entry for "
                f"{class_name}/{original_filename}; found {len(source_rows)}."
            )
        source_relative = _normalize_relative_path(source_rows[0]["filename"])
        if not source_relative.startswith("artaug_input/"):
            raise ValueError(
                f"ArtAug source must point to a training-only copy under artaug_input: "
                f"{source_rows[0]['filename']}"
            )
        source_input = data_dir.joinpath(*PurePosixPath(source_rows[0]["filename"]).parts)
        if not source_input.is_file():
            raise FileNotFoundError(f"Manifest source image is missing: {source_input}")
        if source_input.name.casefold() != original_filename.casefold():
            raise ValueError(
                f"Manifest original_filename does not match artaug_input source for "
                f"{generated_path.name}."
            )
        if _normalize_relative_path(f"{class_name}/{original_filename}") not in original_training_path_set:
            raise ValueError(
                f"ArtAug source {class_name}/{original_filename} is not in the fixed original "
                "seed-42 training split."
            )

        image = load_rgb_array(generated_path)
        samples_by_class[class_name].append((image, row["filename"]))

    counts = {class_name: len(samples_by_class[class_name]) for class_name in CLASSES}
    expected_artaug = len(CLASSES) * EXPECTED_ARTAUG_PER_CLASS
    if sum(counts.values()) != expected_artaug or any(
        count != EXPECTED_ARTAUG_PER_CLASS for count in counts.values()
    ):
        raise ValueError(
            "Expected exactly 10 approved ArtAug images per class "
            f"(30 total); found {counts}."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for label, class_name in enumerate(CLASSES):
        for image, path in sorted(samples_by_class[class_name], key=lambda item: item[1]):
            images.append(image)
            labels.append(label)
            paths.append(path)
    return np.stack(images), np.asarray(labels, dtype=np.int64), paths, counts


def _split_original_with_paths(
    images: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
) -> tuple[np.ndarray, ...]:
    x_train, x_val, x_test, y_train, y_val, y_test, _ = create_split(
        images, labels, paths
    )
    train_val_paths, paths_test, y_train_val_paths, y_test_paths = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    paths_train, paths_val, y_train_paths, y_val_paths = train_test_split(
        train_val_paths,
        y_train_val_paths,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_val_paths,
    )
    if not (
        np.array_equal(y_train, y_train_paths)
        and np.array_equal(y_val, y_val_paths)
        and np.array_equal(y_test, y_test_paths)
        and len(paths_train) == len(x_train)
        and len(paths_val) == len(x_val)
        and len(paths_test) == len(x_test)
    ):
        raise RuntimeError("The path split does not match scripts.train.create_split().")
    return x_train, x_val, x_test, y_train, y_val, y_test, paths_train, paths_val, paths_test


def _assert_original_test_unchanged(paths_test: list[str]) -> None:
    baseline_predictions = RESULTS_DIR.parent / "efficientnetv2b0_with_augmentation_predictions.csv"
    if not baseline_predictions.is_file():
        raise FileNotFoundError(
            f"Cannot verify the original test split; baseline predictions are missing: "
            f"{baseline_predictions}"
        )
    rows = _read_csv(baseline_predictions, {"image_path"})
    saved_test_paths = [row["image_path"] for row in rows]
    if [_normalize_relative_path(path) for path in paths_test] != [
        _normalize_relative_path(path) for path in saved_test_paths
    ]:
        raise RuntimeError(
            "The seed-42 original test split differs from the saved baseline test paths; "
            "refusing to train or evaluate on a potentially changed test split."
        )


def _write_history(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *keys])
        writer.writeheader()
        for epoch_index, loss in enumerate(history["loss"]):
            writer.writerow(
                {
                    "epoch": epoch_index + 1,
                    **{key: values[epoch_index] for key, values in history.items()},
                }
            )


def _write_predictions(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class", "predicted_class", *CLASSES])
        for image_path, true_label, predicted_label, scores in zip(
            paths, y_true, y_pred, probabilities, strict=True
        ):
            writer.writerow(
                [
                    image_path,
                    CLASSES[int(true_label)],
                    CLASSES[int(predicted_label)],
                    *[float(score) for score in scores],
                ]
            )


def _write_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("class", "precision", "recall", "f1-score", "support")
        )
        writer.writeheader()
        for class_name in (*CLASSES, "macro avg", "weighted avg"):
            row = report[class_name]
            writer.writerow(
                {
                    "class": class_name,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1-score": row["f1-score"],
                    "support": int(row["support"]),
                }
            )


def _write_baseline_comparison(artaug_metrics: dict[str, str | int | float]) -> None:
    baseline_path = RESULTS_DIR.parent / "assignment2_metrics.csv"
    baseline_rows = _read_csv(
        baseline_path,
        {
            "model",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        },
    )
    fields = (
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "test_samples",
        "source",
    )
    comparison_path = RESULTS_DIR / "baseline_vs_artaug.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in baseline_rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "source": "Existing measured Assignment 2 baseline",
                }
            )
        writer.writerow(
            {
                **{field: artaug_metrics.get(field, "") for field in fields},
                "source": "Measured ArtAug experiment; same original seed-42 test set",
            }
        )


def prepare_experiment(
    data_dir: Path,
    original_data_dir: Path,
) -> tuple[np.ndarray, ...]:
    original_images, original_labels, original_paths = load_dataset(original_data_dir)
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        paths_train,
        paths_val,
        paths_test,
    ) = _split_original_with_paths(original_images, original_labels, original_paths)
    _assert_original_test_unchanged(paths_test)
    artaug_images, artaug_labels, artaug_paths, artaug_counts = _load_approved_artaug(
        data_dir=data_dir,
        original_training_paths=paths_train,
    )
    x_train_combined = np.concatenate((x_train, artaug_images), axis=0)
    y_train_combined = np.concatenate((y_train, artaug_labels), axis=0)

    if (len(x_train), len(x_val), len(x_test)) != EXPECTED_ORIGINAL_SPLIT:
        raise RuntimeError(
            f"Expected original seed-42 split {EXPECTED_ORIGINAL_SPLIT}, found "
            f"{(len(x_train), len(x_val), len(x_test))}."
        )
    if len(x_train_combined) != 226:
        raise RuntimeError(f"Expected 226 combined training images, found {len(x_train_combined)}.")
    if any("artaug" in path.casefold() for path in (*paths_train, *paths_val, *paths_test)):
        raise RuntimeError("ArtAug image path leaked into an original split.")

    print("Preflight passed:")
    print(f"  Original train: {len(x_train)}")
    print(f"  Approved ArtAug train additions: {len(artaug_images)} ({artaug_counts})")
    print(f"  Combined train: {len(x_train_combined)}")
    print(f"  Original-only validation: {len(x_val)}")
    print(f"  Original-only test: {len(x_test)}")
    print("  Saved-baseline test paths match the seed-42 original split.")
    print(f"  Added ArtAug filenames: {artaug_paths}")
    return (
        x_train_combined,
        y_train_combined,
        x_val,
        y_val,
        x_test,
        y_test,
        paths_test,
        paths_val,
        paths_train,
        artaug_counts,
    )


def _assert_outputs_available(overwrite: bool) -> None:
    existing = [path for path in OUTPUT_FILES if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "ArtAug result files already exist; refusing to overwrite them. "
            "Choose a separate output directory or pass --overwrite after backing up:\n"
            f"{formatted}"
        )


def train(data_dir: Path, original_data_dir: Path, *, check_only: bool, overwrite: bool) -> None:
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        paths_test,
        paths_val,
        paths_train,
        artaug_counts,
    ) = prepare_experiment(data_dir, original_data_dir)
    if check_only:
        print("Check-only mode: no model was created and no training was run.")
        return

    _assert_outputs_available(overwrite)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)

    model = build_classifier(augmentation=True, weights="imagenet")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    metrics = calculate_metrics(
        y_test,
        predictions,
        model_name=EXPERIMENT_NAME,
        augmentation=True,
    )

    model.save(CHECKPOINT_PATH)
    write_metrics_csv([metrics], RESULTS_DIR / "assignment2_metrics.csv")
    _write_history(history.history, RESULTS_DIR / f"{EXPERIMENT_NAME}_history.csv")
    _write_predictions(
        paths_test,
        y_test,
        predictions,
        probabilities,
        RESULTS_DIR / f"{EXPERIMENT_NAME}_predictions.csv",
    )
    _write_classification_report(
        y_test,
        predictions,
        RESULTS_DIR / f"{EXPERIMENT_NAME}_classification_report.csv",
    )
    save_experiment_plots(
        y_test,
        predictions,
        history.history,
        RESULTS_DIR / "figures",
        EXPERIMENT_NAME,
    )
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "seed": SEED,
        "original_train_samples": 196,
        "artaug_train_samples": int(sum(artaug_counts.values())),
        "artaug_train_samples_by_class": artaug_counts,
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "validation_source": "original dataset only",
        "test_source": "original dataset only",
        "class_order": CLASSES,
        "image_size": [128, 128],
        "backbone": "EfficientNetV2B0",
        "weights": "imagenet",
        "backbone_trainable": False,
        "training_augmentation": True,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {"monitor": "val_loss", "patience": EARLY_STOPPING_PATIENCE},
        "test_image_paths": paths_test,
        "validation_image_paths": paths_val,
        "original_train_image_paths": paths_train,
    }
    (RESULTS_DIR / f"{EXPERIMENT_NAME}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_baseline_comparison(metrics)
    print(f"ArtAug checkpoint: {CHECKPOINT_PATH}")
    print(f"ArtAug results: {RESULTS_DIR}")
    print(f"Measured test metrics: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an isolated EfficientNetV2B0 ArtAug experiment; only approved generated "
            "samples are added to the original seed-42 training split."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--original-data-dir", type=Path, default=DEFAULT_ORIGINAL_DATA_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate approvals, files, counts, and fixed test paths without training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only this experiment's separate outputs under artifacts/artaug and results/artaug.",
    )
    args = parser.parse_args()
    train(
        args.data_dir.expanduser().resolve(),
        args.original_data_dir.expanduser().resolve(),
        check_only=args.check_only,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
