from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "comparison.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "model_comparison.png"


def display_name(model: str, status: str) -> str:
    if model.startswith("efficientnetv2b0_"):
        suffix = model.removeprefix("efficientnetv2b0_").replace("_", " ")
        name = f"EfficientNetV2B0 — {suffix}"
    else:
        name = model
    provenance = "A2 measured" if status == "measured" else "A1 reported"
    return f"{name} · {provenance}"


def read_comparison_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Comparison CSV not found: {INPUT_PATH}")

    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 5:
        raise ValueError(f"Expected five models in {INPUT_PATH}, found {len(rows)}.")

    required = {"model", "accuracy", "f1_weighted", "status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain these columns: {', '.join(sorted(required))}")

    labels: list[str] = []
    accuracy: list[float] = []
    weighted_f1: list[float] = []
    for row in rows:
        labels.append(display_name(row["model"], row["status"]))
        try:
            accuracy.append(float(row["accuracy"]))
            weighted_f1.append(float(row["f1_weighted"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid metric value for model '{row['model']}' in {INPUT_PATH}."
            ) from exc

    values = np.asarray([accuracy, weighted_f1], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Accuracy and weighted F1 values must be finite ratios from 0 to 1.")

    return labels, values[0], values[1]


def main() -> None:
    labels, accuracy, weighted_f1 = read_comparison_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#40384D",
            "text.color": "#302A3A",
            "xtick.color": "#514B5B",
            "ytick.color": "#302A3A",
            "savefig.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )

    positions = np.arange(len(labels))
    bar_height = 0.32
    accuracy_color = "#573B83"
    f1_color = "#A58ACB"

    figure, axis = plt.subplots(figsize=(11.2, 6.4), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E9E4F0", linewidth=0.8)
    axis.grid(axis="y", visible=False)

    accuracy_bars = axis.barh(
        positions - bar_height / 2,
        accuracy,
        height=bar_height,
        color=accuracy_color,
        label="Accuracy",
        zorder=3,
    )
    f1_bars = axis.barh(
        positions + bar_height / 2,
        weighted_f1,
        height=bar_height,
        color=f1_color,
        label="Weighted F1-score",
        zorder=3,
    )

    for bars in (accuracy_bars, f1_bars):
        axis.bar_label(
            bars,
            labels=[f"{bar.get_width() * 100:.2f}%" for bar in bars],
            padding=5,
            fontsize=9,
            color="#40384D",
        )

    wrapped_labels = [
        "\n".join(textwrap.wrap(label, width=34, break_long_words=False))
        for label in labels
    ]
    axis.set_yticks(positions, wrapped_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.14)
    axis.set_xticks(np.arange(0, 1.01, 0.2), [f"{tick:.0%}" for tick in np.arange(0, 1.01, 0.2)])
    axis.set_xlabel("Score (%)", labelpad=10)
    axis.set_title("Model comparison", loc="left", fontsize=18, color="#49336F", pad=18)
    axis.text(
        0,
        1.015,
        "Assignment 1 reported baselines and Assignment 2 measured models",
        transform=axis.transAxes,
        fontsize=10,
        color="#70697B",
        va="bottom",
    )
    axis.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(1, 1.01),
        borderaxespad=0,
        columnspacing=1.6,
        handlelength=1.4,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#CFC8D9")
    axis.tick_params(axis="y", length=0, pad=10)
    axis.tick_params(axis="x", length=0, pad=7)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Created chart: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
