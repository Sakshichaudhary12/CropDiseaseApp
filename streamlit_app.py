import streamlit as st
import numpy as np
import os
import json
import gdown
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from remedies import remedies
from gtts import gTTS
from deep_translator import GoogleTranslator

# App title
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
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

# ✅ Updated Prediction Function
def predict(img):
    img = img.convert("RGB")                     # 🔧 Ensure 3 channels (even if grayscale)
    img = img.resize((224, 224))                 # 🔧 Resize for model input
    img_array = image.img_to_array(img) / 255.0  # 🔧 Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # 🔧 Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence

# Upload section
uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        label, confidence = predict(img)
        remedy = remedies.get(label, "No remedy available.")

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence}%**")
        st.warning(f"Recommended Remedy: {remedy}")

        # --- Translate remedy ---
        st.subheader("🌐 Translate Remedy")
        lang_map = {
            'hindi': 'hindi',
            'tamil': 'tamil',
            'bengali': 'bengali',
            'gujarati': 'gujarati',
            'kannada': 'kannada',
            'marathi': 'marathi',
            'telagu': 'telugu',
            'punjabi': 'punjabi'
            'english': 'english'
        }
        lang_code = st.selectbox("Choose Language", list(lang_map.keys()))

        if st.button("Translate"):
            try:
                translated_text = GoogleTranslator(source='auto', target=lang_map[lang_code]).translate(remedy)
                st.write(f"**Translated Remedy ({lang_map[lang_code].capitalize()}):** {translated_text}")
            except Exception as e:
                st.error(f"Translation failed: {e}")

        # --- Voice output ---
        st.subheader("🔊 Listen Remedy")
        if st.button("Speak Remedy"):
            try:
                tts = gTTS(text=remedy, lang='en')
                tts.save("remedy.mp3")
                audio_file = open("remedy.mp3", "rb")
                st.audio(audio_file.read(), format="audio/mp3")
            except Exception as e:
                st.error(f"Text-to-Speech failed: {e}")

        # --- Feedback section ---
        st.subheader("📝 Feedback")
        feedback = st.radio("Was the prediction correct?", ("Yes", "No"))
        comment = st.text_input("Any suggestions or comments?")

        if st.button("Submit Feedback"):
            feedback_data = {
                "image_name": uploaded_file.name,
                "prediction": label,
                "confidence": confidence,
                "feedback": feedback,
                "comment": comment
            }
            df = pd.DataFrame([feedback_data])
            if os.path.exists("feedback_log.csv"):
                df.to_csv("feedback_log.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("feedback_log.csv", index=False)
            st.success("Thank you for your feedback!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
