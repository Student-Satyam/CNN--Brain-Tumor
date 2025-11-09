# ================================================
# 🧠 Brain Tumor Detection Streamlit App
# ================================================

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, login

# ================================================
# Hugging Face Authentication (for Streamlit Cloud)
# ================================================
# You will add your HF_TOKEN in Streamlit Cloud → Settings → Secrets
# Example:
# HF_TOKEN = "your_huggingface_access_token"
login(token=os.getenv("HF_TOKEN"))

# ================================================
# Load the saved CNN model from Hugging Face
# ================================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Student-Satyam/brain-tumor-cnn",
        filename="brain_tumor_cnn_model.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ================================================
# Streamlit UI
# ================================================
st.title("🧠 Brain Tumor Detection using CNN by Satyam")

# Educational disclaimer
st.warning(
    "⚠️ **Disclaimer:** This app is for **educational and research purposes only**. "
    "It is **NOT a medical diagnostic tool**. "
    "Predictions are based on an AI model and may be inaccurate. "
    "Always consult a qualified doctor for medical concerns."
)

st.write("Upload an MRI image to check whether it **may indicate** a brain tumor or not.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded MRI Image", use_container_width=True)

    # Preprocess image for model prediction
    img = image.resize((150, 150))  # must match model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])

    if probability > 0.5:
        result_text = "🧠 Tumor Detected (AI Prediction)"
    else:
        result_text = "✅ No Tumor Detected (AI Prediction)"

    # Output results
    st.subheader("Prediction Result")
    st.success(result_text)
    st.info(
        f"**Note:** This result is for **educational demonstration only**. "
        f"AI model confidence: {probability:.2f}. "
        f"Always consult a medical professional."
    )

# ================================================
# About Section
# ================================================
with st.expander("ℹ️ About this App"):
    st.write("""
    This app uses a **Convolutional Neural Network (CNN)** model trained on MRI images
    to demonstrate how AI can be applied in medical image classification.

    **Important Notes:**
    - This is for **learning and research** purposes only.
    - It is **not intended for real clinical diagnosis**.
    - Real-world deployment would require proper medical validation,
      data compliance, and expert supervision.
    """)
