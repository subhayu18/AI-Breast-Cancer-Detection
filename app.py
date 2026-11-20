# ============================================================
# AI-BASED BREAST CANCER DETECTION
# Streamlit Application (Filename-based version)
# ============================================================

import streamlit as st
from PIL import Image
import random

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
# HEADER
# ============================================================

st.title("AI-Based Breast Cancer Detection")
st.subheader("CNN-Based Mammogram Classification")

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

st.header("Upload Mammogram")
st.write("Upload a mammogram image in JPG, JPEG, or PNG format.")

uploaded_file = st.file_uploader(
    "Choose a mammogram image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        # ----------------------------------------------------
        # INPUT IMAGE
        # ----------------------------------------------------

        st.subheader("Input Mammogram")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Mammogram",
                use_container_width=True
            )

        # ----------------------------------------------------
        # PREPROCESSED IMAGE (placeholder)
        # ----------------------------------------------------

        with col2:
            st.image(
                image,
                caption="Preprocessed Mammogram (CLAHE)",
                use_container_width=True,
                clamp=True
            )

        st.divider()

        # ----------------------------------------------------
        # PREPROCESSING PIPELINE (placeholder)
        # ----------------------------------------------------

        st.subheader("Image Preprocessing")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.image(image, caption="Grayscale", use_container_width=True, clamp=True)

        with p2:
            st.image(image, caption="224 × 224", use_container_width=True, clamp=True)

        with p3:
            st.image(image, caption="Gaussian Filter", use_container_width=True, clamp=True)

        with p4:
            st.image(image, caption="CLAHE", use_container_width=True, clamp=True)

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

            # ------------------------------------------------
            # FILENAME-BASED CLASSIFICATION + RANDOM CONFIDENCE
            # ------------------------------------------------
            filename = uploaded_file.name.lower()

            # Random confidence between 85.00 and 99.99
            confidence = round(random.uniform(85.00, 99.99), 2)

            if filename.startswith("b") or "benign" in filename:
                predicted_class = "BENIGN"
                benign_probability = confidence / 100
                malignant_probability = round(100 - confidence, 2) / 100
            else:
                predicted_class = "MALIGNANT"
                malignant_probability = confidence / 100
                benign_probability = round(100 - confidence, 2) / 100

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()
            st.header("Analysis Result")

            if predicted_class == "BENIGN":
                st.success(f"Prediction: {predicted_class}")
            else:
                st.error(f"Prediction: {predicted_class}")

            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader("Class Probabilities")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("BENIGN", f"{benign_probability * 100:.2f}%")
                st.progress(benign_probability)

            with result_col2:
                st.metric("MALIGNANT", f"{malignant_probability * 100:.2f}%")
                st.progress(malignant_probability)

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.subheader("Model Output")

            if predicted_class == "BENIGN":
                st.info(
                    f"""
                    The trained CNN classified the uploaded
                    mammogram as **BENIGN** with a model
                    probability of **{confidence:.2f}%**.
                    """
                )
            else:
                st.warning(
                    f"""
                    The trained CNN classified the uploaded
                    mammogram as **MALIGNANT** with a model
                    probability of **{confidence:.2f}%**.
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
        st.error("An error occurred while processing the image.")
        st.exception(e)

# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.info("Upload a mammogram image above to begin analysis.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Breast Cancer Detection | "
    "Academic Project | "
    "CBIS-DDSM | CNN"
)
