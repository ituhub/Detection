# app.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from models import load_xray_model, load_ecg_model

# Load models
xray_model = load_xray_model()
ecg_model = load_ecg_model()
# Load other models as needed

# Set up the app interface
st.title("Disease Detection and Prediction System")

# Sidebar for report selection
report_type = st.sidebar.selectbox(
    "Select the type of medical report:",
    ("X-ray", "Ultrasound", "ECG", "CT Scan", "MRI")
)

# File uploader
uploaded_file = st.file_uploader("Upload Medical Report", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Report', use_column_width=True)

    # Preprocess the image
    def preprocess_image(img, target_size):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.asarray(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    if report_type == "X-ray":
        preprocessed_image = preprocess_image(image, target_size=(224, 224))
        prediction = xray_model.predict(preprocessed_image)
        # Interpret prediction
        if prediction[0][0] > 0.5:
            st.success("Prediction: Positive for Disease")
        else:
            st.info("Prediction: Negative for Disease")

    elif report_type == "ECG":
        # Additional preprocessing for ECG if needed
        preprocessed_image = preprocess_image(image, target_size=(224, 224))
        prediction = ecg_model.predict(preprocessed_image)
        # Interpret prediction
        if prediction[0][0] > 0.5:
            st.success("Prediction: Abnormal ECG")
        else:
            st.info("Prediction: Normal ECG")

    # Add similar blocks for Ultrasound, CT Scan, MRI

else:
    st.warning("Please upload a medical report to proceed.")
