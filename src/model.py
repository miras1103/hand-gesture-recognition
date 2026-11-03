from __future__ import annotations

import tensorflow as tf

from src.preprocessing import IMAGE_SIZE, CLASSES


def build_classifier(
    *,
    augmentation: bool,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build an EfficientNetV2B0 transfer-learning classifier."""
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32,
        name="rgb_image_0_255",
    )
    x = inputs

    if augmentation:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ],
            name="training_augmentation",
        )(x)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_preprocessing=True,
    )
    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=backbone.output,
        name="backbone",
    )
    feature_extractor.trainable = False
    features = feature_extractor(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    x = tf.keras.layers.Dropout(0.25, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASSES), activation="softmax", name="class_probabilities"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gesture_efficientnetv2b0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
