from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.explainability import make_gradcam_heatmap, overlay_gradcam
from src.preprocessing import CLASSES, IMAGE_SIZE


def predict(
    model: tf.keras.Model,
    image_rgb: np.ndarray,
    *,
    latency_repeats: int = 5,
    display_image_rgb: np.ndarray | None = None,
) -> tuple[int, np.ndarray, float, np.ndarray]:
    """Predict classes, average inference latency, and a Grad-CAM overlay."""
    resized = tf.image.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
    ).numpy().astype(np.float32)
    batch = np.expand_dims(resized, axis=0)

    model(batch, training=False)
    start = time.perf_counter()
    for _ in range(latency_repeats):
        probabilities = model(batch, training=False).numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000.0 / latency_repeats

    predicted_index = int(np.argmax(probabilities))
    if len(probabilities) != len(CLASSES):
        raise ValueError(
            f"Model returned {len(probabilities)} scores; expected {len(CLASSES)} classes."
        )
    heatmap = make_gradcam_heatmap(model, batch, predicted_index)
    overlay_image = image_rgb if display_image_rgb is None else display_image_rgb
    overlay = overlay_gradcam(overlay_image, heatmap)
    return predicted_index, probabilities, latency_ms, overlay
