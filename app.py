from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "artifacts" / "efficientnetv2b0_with_augmentation.keras"
MODEL_PATH = Path(os.environ.get("GESTURE_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
CLASS_NAMES = ("peace", "stop", "thumbs_up")

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #ffffff; color: #29243a; }
    h1, h2, h3 { color: #4b347c; }
    .block-container {
        box-sizing: border-box;
        max-width: 1160px;
        margin: 1.5rem auto;
        padding: 1.5rem 2rem 1rem;
        background: #f5f1fb;
        border: 1px solid #dcd2eb;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(49, 35, 75, 0.09);
    }
    @media (max-width: 700px) {
        .block-container {
            margin: 0.75rem auto;
            padding: 1rem 0.85rem;
            border-radius: 12px;
        }
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #e8e2f2 !important;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(49, 35, 75, 0.04);
    }
    h3 {
        margin-top: 0.6rem;
        margin-bottom: 0.35rem;
    }
    .stButton > button, .stDownloadButton > button {
        background: #563b8c; color: #ffffff; border: 0; border-radius: 9px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #432d70; color: #ffffff; border-color: #432d70;
    }
    [data-testid="stMetric"] {
        background: #f0ecfa; border: 1px solid #e1daf3;
        padding: 14px; border-radius: 10px;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #29243a !important;
    }
    [data-testid="stProgress"] [role="progressbar"] {
        box-sizing: border-box;
        background: #ffffff !important;
        border: 1px solid #c9bce3;
        border-radius: 999px;
        overflow: hidden;
    }
    [data-testid="stProgress"] [role="progressbar"] > div {
        background: #7658aa !important;
        border-radius: 999px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 1rem;
    }
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"],
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] * {
        color: #29243a !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label * {
        color: #29243a !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        border-color: #563b8c !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div:first-child {
        border-color: #563b8c !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked)
        > div:first-child > div {
        background-color: #563b8c !important;
    }
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stCheckbox"] input[type="checkbox"] {
        accent-color: #563b8c !important;
    }
    [data-testid="stFileUploader"] button {
        background: #563b8c !important;
        color: #ffffff !important;
        border: 1px solid #563b8c !important;
        border-radius: 9px;
    }
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:hover *,
    [data-testid="stFileUploader"] button:focus-visible,
    [data-testid="stFileUploader"] button:focus-visible * {
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #432d70 !important;
        border-color: #432d70 !important;
    }
    [data-testid="stFileUploader"] button:focus-visible {
        outline: 3px solid #b9a5df !important;
        outline-offset: 2px;
    }
    [data-testid="stFileUploader"] {
        box-sizing: border-box;
        width: 100%;
        max-width: 520px;
        margin: 0 auto;
        padding: 12px 16px;
        background: #f0ecfa;
        border: 0;
        border-radius: 14px;
    }
    [data-testid="stFileUploaderDropzone"] {
        box-sizing: border-box;
        width: 100%;
        min-height: 104px !important;
        height: 104px !important;
        padding: 0.5rem 0.75rem !important;
        background: transparent !important;
        border: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        gap: 0.35rem !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-size: 0.85rem;
        color: #393149 !important;
    }
    [data-testid="stFileUploader"] :not(button) {
        color: #29243a !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] * {
        color: #29243a !important;
    }
    [data-testid="stFileUploader"] svg {
        color: #563b8c !important;
    }
    [data-testid="stFileUploader"] button[aria-label*="Remove"],
    [data-testid="stFileUploader"] button[title*="Remove"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
        box-sizing: border-box;
        width: 2.25rem;
        min-width: 2.25rem;
        height: 2.25rem;
        padding: 0.4rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #563b8c !important;
        color: #ffffff !important;
        border: 1px solid #563b8c !important;
        border-radius: 9px;
    }
    [data-testid="stFileUploader"] button[aria-label*="Remove"] svg,
    [data-testid="stFileUploader"] button[title*="Remove"] svg,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg {
        filter: brightness(0) invert(1);
    }
    [data-testid="stFileUploader"] button[aria-label*="Remove"] *,
    [data-testid="stFileUploader"] button[title*="Remove"] *,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] * {
        color: #ffffff !important;
    }
    div[data-testid="stImage"] {
        box-sizing: border-box;
        padding: 8px;
        background: #ffffff;
        border: 1px solid #e8e2f2;
        border-radius: 12px;
    }
    div[data-testid="stCameraInput"] {
        width: 100%;
        max-width: 380px;
        margin-inline: auto;
        padding: 0;
        overflow: hidden;
    }
    div[data-testid="stCameraInput"] > div,
    div[data-testid="stCameraInput"] [data-testid="stCameraInputWebcamComponent"] {
        width: 100% !important;
        max-width: 380px;
        margin: 0 auto !important;
        padding: 0 !important;
        background: transparent !important;
    }
    div[data-testid="stCameraInputWebcamStyledBox"] {
        width: 100% !important;
        max-width: 380px;
        height: auto !important;
        min-height: 0;
        aspect-ratio: auto !important;
        margin-inline: auto;
        background: transparent !important;
        overflow: hidden !important;
    }
    div[data-testid="stCameraInput"] img,
    div[data-testid="stCameraInputWebcamStyledBox"] video {
        display: block;
        width: 100%;
        max-width: 380px;
        height: auto !important;
        max-height: none;
        margin-inline: auto;
        object-fit: contain;
        aspect-ratio: auto !important;
    }
    div[data-testid="stCameraInput"] p {
        margin-bottom: 0.35rem;
    }
    div[data-testid="stCameraInput"] button {
        display: flex;
        width: fit-content;
        min-width: 180px;
        margin: 0.35rem auto 0;
        background: #563b8c !important;
        color: #ffffff !important;
        border: 0;
        border-radius: 9px;
    }
    div[data-testid="stCameraInput"] button:hover {
        background: #432d70 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hand Gesture Recognition")
st.caption("Computer Vision · Assignment 2")
st.write(
    "Upload a hand image or capture one with your camera to classify "
    "peace, stop, or thumbs up."
)

if not MODEL_PATH.is_file():
    st.error(
        "The trained model file was not found. Train the model first with "
        "`python scripts\\train.py`, or set the `GESTURE_MODEL_PATH` environment variable."
    )
    st.code(str(MODEL_PATH), language=None)
    st.stop()

try:
    import tensorflow as tf

    from src.inference import predict
    from src.preprocessing import load_rgb_array, open_oriented_rgb_image
except ImportError as exc:
    st.error(
        "A required application dependency is missing. Activate the project "
        "virtual environment and install `requirements.txt`."
    )
    st.exception(exc)
    st.stop()


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


with st.container(border=True):
    st.subheader("1. Image input")
    source = st.radio(
        "Image source",
        ("Upload an image", "Use camera"),
        horizontal=True,
        label_visibility="visible",
    )
    image_file = (
        st.file_uploader(
            "Choose a JPG, JPEG, PNG, or WEBP image",
            type=("jpg", "jpeg", "png", "webp"),
            key="uploaded_image",
        )
        if source == "Upload an image"
        else st.camera_input("Capture a hand image")
    )

if image_file is not None:
    try:
        input_image = open_oriented_rgb_image(image_file)
        image_array = load_rgb_array(input_image)
        display_image = input_image.copy()
        display_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        display_image = np.asarray(display_image)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read this image. Please choose a valid image file. Details: {exc}")
        st.stop()

    try:
        model = load_model(str(MODEL_PATH.resolve()))
        predicted_index, scores, latency_ms, overlay = predict(
            model,
            image_array,
            display_image_rgb=display_image,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        tf.errors.OpError,
    ) as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("2. Prediction results")
    input_image_col, result_col = st.columns([1, 1], gap="medium")
    with input_image_col:
        st.image(display_image, caption="Input image", width=380)

    with result_col:
        st.metric("Predicted gesture", CLASS_NAMES[predicted_index])
        st.caption(
            "Scores are model probabilities and should not be interpreted as calibrated certainty."
        )
        st.markdown("#### Class probabilities")
        for class_name, probability in zip(CLASS_NAMES, scores, strict=True):
            st.write(f"**{class_name}** · {probability:.1%}")
            st.progress(float(probability))

    explanation_col, performance_col = st.columns([1, 1], gap="medium")
    with explanation_col:
        st.subheader("3. Grad-CAM explanation")
        st.image(
            overlay,
            caption="Grad-CAM overlay — highlights image regions contributing to the selected class.",
            width=380,
        )
        st.caption(
            "Grad-CAM is an approximate visual explanation, not proof that the model reasons correctly."
        )

    with performance_col:
        st.subheader("4. Inference performance")
        st.metric(
            "Average inference latency",
            f"{latency_ms:.2f} ms",
            help="Mean of 5 predictions after one warm-up call; excludes Grad-CAM.",
        )
        st.subheader("Environment")
        logical_devices = tf.config.list_logical_devices()
        device_names = sorted({device.device_type for device in logical_devices})
        st.write(f"**Python:** {platform.python_version()}")
        st.write(f"**TensorFlow device(s):** {', '.join(device_names) or 'Unavailable'}")
        try:
            browser = st.context.headers.get("User-Agent", "Unavailable")
        except (AttributeError, KeyError):
            browser = "Unavailable"
        st.write(f"**Browser:** {browser}")

st.divider()
st.caption(
    "Images are processed by the Streamlit server for prediction and are not saved by this application."
)
