# ================================================
# 🧠 Brain Tumor Detection Streamlit App
# ================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================================================
# Load the saved CNN model
# ================================================
@st.cache_resource
def load_model():
    # Load your model (you saved it as .keras)
    model = tf.keras.models.load_model("brain_tumor_cnn_model.keras")
    return model

model = load_model()

# ================================================
# Streamlit UI
# ================================================
st.title("🧠 Brain Tumor Detection using CNN by Satyam")

# Educational disclaimer
st.warning(
    "⚠️ **Disclaimer:** This app is for **educational purposes only**. "
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

    # Output
    st.subheader("Prediction Result")
    st.success(result_text)
    st.info(
        f"**Note:** This result is for **educational purposes** and may be inaccurate. "
        f"AI prediction confidence: {probability:.2f}. Always consult a doctor."
    )

# Optional: About section
with st.expander("ℹ️ About this App"):
    st.write("""
    This app uses a Convolutional Neural Network (CNN) to predict whether an MRI image may indicate a brain tumor.
    
    **Important:**
    - This app is for learning and research purposes.
    - Do not rely on this for medical diagnosis.
    - Always consult a qualified healthcare professional for medical advice.
    """)
