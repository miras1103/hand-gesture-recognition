from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.experiments import calculate_metrics, save_experiment_plots, write_metrics_csv
from src.preprocessing import CLASSES, load_dataset
from scripts.train import SEED, create_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the fixed stratified test split."
    )
    parser.add_argument("model", type=Path, help="Path to a saved .keras model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORKSPACE_DIR / "dataset",
        help="Dataset folder containing the three class subfolders",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file does not exist: {args.model}")

    images, labels, paths = load_dataset(args.data_dir)
    _, _, x_test, _, _, y_test, paths_test = create_split(images, labels, paths)
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    metrics = calculate_metrics(
        y_test,
        predictions,
        model_name=args.model.stem,
        augmentation="with_augmentation" in args.model.stem,
    )

    output_dir = PROJECT_DIR / "results"
    save_experiment_plots(
        y_test,
        predictions,
        {},
        output_dir / "figures",
        args.model.stem,
    )
    write_metrics_csv([metrics], output_dir / f"{args.model.stem}_evaluation.csv")
    with (output_dir / f"{args.model.stem}_evaluation_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class", "predicted_class", *CLASSES])
        for image_path, true_label, predicted_label, scores in zip(
            paths_test, y_test, predictions, probabilities, strict=True
        ):
            writer.writerow(
                [
                    image_path,
                    CLASSES[int(true_label)],
                    CLASSES[int(predicted_label)],
                    *[float(score) for score in scores],
                ]
            )
    print(metrics)
    print(f"Evaluated with seed={SEED} on {len(y_test)} held-out test samples.")


if __name__ == "__main__":
    main()
