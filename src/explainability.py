from __future__ import annotations

import numpy as np
import tensorflow as tf
from matplotlib import colormaps


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Create Grad-CAM from EfficientNetV2B0's final spatial feature map."""
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        try:
            augmentation = model.get_layer("training_augmentation")
            inputs = augmentation(inputs, training=False)
        except ValueError:
            pass
        feature_maps = model.get_layer("backbone")(inputs, training=False)
        tape.watch(feature_maps)
        pooled = model.get_layer("global_average_pooling")(feature_maps)
        dropped = model.get_layer("classifier_dropout")(pooled, training=False)
        probabilities = model.get_layer("class_probabilities")(dropped)
        class_score = probabilities[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be computed for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(channel_weights * feature_maps, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)
    return heatmap[0].numpy()


def overlay_gradcam(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a normalized heatmap onto an RGB image."""
    image = np.asarray(image_rgb)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Grad-CAM overlay requires an RGB image with shape (height, width, 3).")
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Grad-CAM heatmap must be a finite 2D array.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Grad-CAM overlay alpha must be between 0 and 1.")

    image = np.clip(image, 0, 255).astype(np.uint8)
    heat = tf.image.resize(
        heatmap[..., np.newaxis],
        (image.shape[0], image.shape[1]),
        method="bilinear",
    ).numpy()[..., 0]
    heat = np.maximum(heat, 0.0)
    peak = float(np.max(heat))
    if peak == 0.0:
        return image

    positive = heat[heat > 0.0]
    scale = float(np.percentile(positive, 99))
    if scale <= 0.0:
        return image
    intensity = np.clip(heat / scale, 0.0, 1.0) ** 0.65
    heat_rgb = colormaps["turbo"](intensity)[..., :3] * 255.0
    heat_alpha = alpha * intensity[..., np.newaxis]
    blended = (1.0 - heat_alpha) * image.astype(np.float32) + heat_alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)
