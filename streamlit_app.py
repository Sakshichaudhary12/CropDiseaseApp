import streamlit as st
import numpy as np
import os
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from remedies import remedies

# Try to import gdown, install if missing
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# Title
st.title("🌿 Crop Disease Detector")
st.write("Upload a sugarcane or maize leaf image to detect disease and get treatment advice.")

# Step 1: Download model from Google Drive if not present
model_path = "crop_disease_model.h5"
if not os.path.exists(model_path):
    file_id = "1erkVOB1_fsbO2H8SOguACLuKpPO3ehhb"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Step 2: Load model
model = load_model(model_path)

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Prediction function
def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(img)
    remedy = remedies.get(label, "No remedy available.")

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence}%")
    st.warning(f"Recommended Remedy: {remedy}")
