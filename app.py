# ============================================================
# AI-BASED BREAST CANCER DETECTION
# Streamlit Application
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "subhayu18/ai-breast-cancer-cnn"
HF_MODEL_FILENAME = "breast_cancer_cnn.keras"

BASE_DIR = Path(__file__).resolve().parent

CLASS_NAMES_PATH = (
    BASE_DIR / "model" / "class_names.json"
)

CONFIG_PATH = (
    BASE_DIR / "model" / "preprocessing_config.json"
)


# ============================================================
# DOWNLOAD / LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with st.spinner(
        "Loading the trained CNN model..."
    ):

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME
        )

        loaded_model = tf.keras.models.load_model(
            model_path
        )

    return loaded_model


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_configuration():

    class_names = [
        "BENIGN",
        "MALIGNANT"
    ]

    config = {
        "image_size": [224, 224],
        "normalization": "pixel_value / 255.0",
        "gaussian_blur": {
            "kernel": [5, 5],
            "sigma": 0
        },
        "clahe": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8]
        }
    }

    if CLASS_NAMES_PATH.exists():

        with open(
            CLASS_NAMES_PATH,
            "r"
        ) as f:

            class_names = json.load(f)

    if CONFIG_PATH.exists():

        with open(
            CONFIG_PATH,
            "r"
        ) as f:

            config = json.load(f)

    return class_names, config


# ============================================================
# LOAD MODEL + CONFIGURATION
# ============================================================

model = load_model()

class_names, config = load_configuration()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    """
    Applies the same preprocessing used during training:

    1. Grayscale conversion
    2. Resize to 224 × 224
    3. Gaussian filtering
    4. CLAHE enhancement
    5. Normalization
    6. Conversion to 3 channels
    """

    image_array = np.array(image)

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    if len(image_array.shape) == 3:

        if image_array.shape[2] == 4:

            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGBA2RGB
            )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_array

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Gaussian filtering
    # --------------------------------------------------------

    gaussian = cv2.GaussianBlur(
        resized,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gaussian
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = (
        enhanced.astype(np.float32) / 255.0
    )

    # --------------------------------------------------------
    # Grayscale → 3 channels
    # --------------------------------------------------------

    rgb = np.stack(
        [
            normalized,
            normalized,
            normalized
        ],
        axis=-1
    )

    # Batch dimension
    model_input = np.expand_dims(
        rgb,
        axis=0
    )

    return (
        gray,
        resized,
        gaussian,
        enhanced,
        model_input
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI-Based Breast Cancer Detection"
)

st.subheader(
    "CNN-Based Mammogram Classification"
)

st.markdown(
    """
An academic computer-aided classification system for
mammogram images using a Convolutional Neural Network (CNN)
trained on the CBIS-DDSM dataset.
"""
)

st.divider()


# ============================================================
# SIDEBAR — MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.markdown(
        """
        **Dataset**

        CBIS-DDSM

        **Model**

        Custom Convolutional Neural Network

        **Input**

        224 × 224 × 3

        **Classes**

        - BENIGN
        - MALIGNANT

        **Preprocessing**

        - Grayscale conversion
        - Resize
        - Gaussian filtering
        - CLAHE
        - Normalization

        **Optimizer**

        Adam

        **Learning Rate**

        0.001

        **Batch Size**

        32

        **Maximum Epochs**

        50
        """
    )

    st.divider()

    st.header("CNN Architecture")

    st.markdown(
        """
        **Input**

        224 × 224 × 3

        ↓

        **Conv2D**

        32 filters, 3 × 3

        ↓

        **MaxPooling**

        ↓

        **Conv2D**

        64 filters, 3 × 3

        ↓

        **MaxPooling**

        ↓

        **Conv2D**

        128 filters, 3 × 3

        ↓

        **MaxPooling**

        ↓

        **Flatten**

        ↓

        **Dense**

        128 neurons

        ↓

        **Dropout**

        0.5

        ↓

        **Softmax**

        2 classes
        """
    )

    st.divider()

    st.caption(
        "Academic research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header(
    "Upload Mammogram"
)

st.write(
    "Upload a mammogram image in JPG, JPEG, or PNG format."
)

uploaded_file = st.file_uploader(
    "Choose a mammogram image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        )

        # ----------------------------------------------------
        # INPUT IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Input Mammogram"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Mammogram",
                use_container_width=True
            )

        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        (
            gray,
            resized,
            gaussian,
            enhanced,
            model_input
        ) = preprocess_image(
            image
        )

        # ----------------------------------------------------
        # PREPROCESSED IMAGE
        # ----------------------------------------------------

        with col2:

            st.image(
                enhanced,
                caption="Preprocessed Mammogram (CLAHE)",
                use_container_width=True,
                clamp=True
            )

        st.divider()

        # ----------------------------------------------------
        # PREPROCESSING PIPELINE
        # ----------------------------------------------------

        st.subheader(
            "Image Preprocessing"
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:

            st.image(
                gray,
                caption="Grayscale",
                use_container_width=True,
                clamp=True
            )

        with p2:

            st.image(
                resized,
                caption="224 × 224",
                use_container_width=True,
                clamp=True
            )

        with p3:

            st.image(
                gaussian,
                caption="Gaussian Filter",
                use_container_width=True,
                clamp=True
            )

        with p4:

            st.image(
                enhanced,
                caption="CLAHE",
                use_container_width=True,
                clamp=True
            )

        st.divider()

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        analyze = st.button(
            "🔍 ANALYZE MAMMOGRAM",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Analyzing mammogram..."
            ):

                probabilities = model.predict(
                    model_input,
                    verbose=0
                )[0]

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                probabilities[predicted_index]
            )

            benign_probability = float(
                probabilities[0]
            )

            malignant_probability = float(
                probabilities[1]
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.header(
                "Analysis Result"
            )

            if predicted_class == "BENIGN":

                st.success(
                    f"Prediction: {predicted_class}"
                )

            else:

                st.error(
                    f"Prediction: {predicted_class}"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "BENIGN",
                    f"{benign_probability * 100:.2f}%"
                )

                st.progress(
                    benign_probability
                )

            with result_col2:

                st.metric(
                    "MALIGNANT",
                    f"{malignant_probability * 100:.2f}%"
                )

                st.progress(
                    malignant_probability
                )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.subheader(
                "Model Output"
            )

            if predicted_class == "BENIGN":

                st.info(
                    f"""
                    The trained CNN classified the uploaded
                    mammogram as **BENIGN** with a model
                    probability of **{confidence * 100:.2f}%**.
                    """
                )

            else:

                st.warning(
                    f"""
                    The trained CNN classified the uploaded
                    mammogram as **MALIGNANT** with a model
                    probability of **{confidence * 100:.2f}%**.
                    """
                )

            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.divider()

            st.warning(
                """
                **Research Disclaimer**

                This application is an academic computer-aided
                classification prototype developed for research
                and educational purposes.

                The prediction is generated by a machine-learning
                model and must not be interpreted as a medical
                diagnosis. The system has not been clinically
                validated or approved for clinical decision-making.
                """
            )

    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload a mammogram image above to begin analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Breast Cancer Detection | "
    "Academic Project | "
    "CBIS-DDSM | CNN"
)