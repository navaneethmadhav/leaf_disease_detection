# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# ========== Load Model & Labels ==========
model = tf.keras.models.load_model('model/plant_disease_model.h5')

with open('model/labels.json') as f:
    labels = json.load(f)
label_map = {int(k): v for k, v in labels.items()}

# ========== Streamlit UI ==========
st.title("🌿 Plant Disease Detection using CNN")
st.write("Upload a leaf image to check if it is healthy or unhealthy.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Leaf Image', use_container_width=True)

    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = label_map[pred_index]

    st.subheader("🔍 Prediction Result:")
    if "healthy" in pred_class.lower():
        st.success(f"✅ The leaf is **Healthy**")
    else:
        st.error(f"⚠️ The leaf is **Unhealthy**")
