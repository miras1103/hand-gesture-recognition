from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "assignment2_metrics.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "model_metrics_comparison.png"

METRICS = (
    ("accuracy", "Accuracy"),
    ("precision_macro", "Macro Precision"),
    ("recall_macro", "Macro Recall"),
    ("f1_macro", "Macro F1-score"),
    ("precision_weighted", "Weighted Precision"),
    ("recall_weighted", "Weighted Recall"),
    ("f1_weighted", "Weighted F1-score"),
)
MODEL_NAMES = {
    "efficientnetv2b0_with_augmentation": "EfficientNetV2B0 with augmentation",
    "efficientnetv2b0_without_augmentation": "EfficientNetV2B0 without augmentation",
}


def read_metrics() -> tuple[list[str], np.ndarray]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {INPUT_PATH}")

    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required_fields = {"model", *(field for field, _ in METRICS)}
    if len(rows) != len(MODEL_NAMES) or not rows or not required_fields.issubset(rows[0]):
        raise ValueError(
            f"{INPUT_PATH} must contain both EfficientNetV2B0 rows and fields "
            f"{', '.join(sorted(required_fields))}."
        )

    rows_by_model = {row["model"]: row for row in rows}
    if set(rows_by_model) != set(MODEL_NAMES):
        raise ValueError(
            f"Unexpected model rows in {INPUT_PATH}: {', '.join(sorted(rows_by_model))}."
        )

    labels = [MODEL_NAMES[model] for model in MODEL_NAMES]
    try:
        values = np.asarray(
            [
                [float(rows_by_model[model][field]) for field, _ in METRICS]
                for model in MODEL_NAMES
            ],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metrics in {INPUT_PATH} must be numeric.") from exc

    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Metrics in {INPUT_PATH} must be finite ratios between 0 and 1.")

    return labels, values


def main() -> None:
    labels, values = read_metrics()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#40384D",
            "text.color": "#302A3A",
            "xtick.color": "#514B5B",
            "ytick.color": "#514B5B",
            "savefig.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )

    positions = np.arange(len(METRICS))
    bar_width = 0.36
    colors = ("#573B83", "#A58ACB")

    figure, axis = plt.subplots(figsize=(12, 6.6), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E9E4F0", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)

    for model_index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        bars = axis.bar(
            positions + (model_index - 0.5) * bar_width,
            values[model_index],
            width=bar_width,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2%}" for value in values[model_index]],
            padding=3,
            fontsize=8,
            color="#40384D",
        )

    axis.set_xticks(positions, [label for _, label in METRICS], rotation=12, ha="right")
    axis.set_ylim(0, 1.12)
    axis.set_yticks(np.arange(0, 1.01, 0.2), [f"{value:.0%}" for value in np.arange(0, 1.01, 0.2)])
    axis.set_ylabel("Score (%)", labelpad=10)
    axis.set_title("Model Evaluation Metrics", loc="left", fontsize=18, color="#49336F", pad=16)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_color("#CFC8D9")
    axis.spines["bottom"].set_color("#CFC8D9")
    axis.tick_params(axis="both", length=0, pad=7)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Created chart: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
