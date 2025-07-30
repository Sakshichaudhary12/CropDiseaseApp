import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
from gtts import gTTS
import os
from googletrans import Translator

# Remedies dictionary
remedies = {
    'maize_Blight': "Use resistant varieties and avoid overhead irrigation.",
    'maize_Gray_leaf_Spot': "Apply fungicides and practice crop rotation.",
    'maize_rust': "Use resistant hybrids and fungicide spray.",
    'maize_healthy': "No disease detected. Crop is healthy.",
    'Sugarcane_Mosaic': "Use certified virus-free planting material.",
    'Sugarcane_RedRot': "Remove and destroy affected clumps. Use resistant varieties.",
    'Sugarcane_Rust': "Apply recommended fungicides and burn infected leaves.",
    'Sugarcane_Yellow': "Improve drainage and use fungicide where necessary.",
    'Sugarcane_Healthy': "No disease detected. Crop is healthy."
}

# Load model and class labels
model = load_model("crop_disease_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Translator
translator = Translator()

def translate_remedy(text, lang='hi'):
    try:
        return translator.translate(text, dest=lang).text
    except:
        return "Translation failed."

def speak_remedy(text, lang='hi'):
    tts = gTTS(text, lang=lang)
    tts.save("remedy.mp3")
    audio_file = open("remedy.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# Streamlit UI
st.title("🌾 Crop Disease Detector")
st.write("Upload a **sugarcane or maize leaf** image for disease classification and treatment guidance.")

uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

def predict(img):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        predicted_class = class_labels.get(class_index, "Unknown")
        confidence = round(100 * float(np.max(prediction)), 2)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return "Error", 0.0

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    label, confidence = predict(img)
    
    if label != "Error":
        st.success(f"🧠 Prediction: {label}")
        st.info(f"🔍 Confidence: {confidence}%")

        remedy = remedies.get(label, "No remedy available.")
        st.warning(f"💊 Recommended Remedy: {remedy}")

        translated = translate_remedy(remedy, lang='hi')
        st.markdown(f"🌐 **Translated Remedy (Hindi):** {translated}")
        speak_remedy(translated)

        feedback = st.radio("📝 Was this prediction accurate?", ("Yes", "No"))
        if st.button("Submit Feedback"):
            with open("feedback_log.csv", "a") as f:
                f.write(f"{label},{confidence},{feedback}\n")
            st.success("✅ Feedback saved. Thank you!")

