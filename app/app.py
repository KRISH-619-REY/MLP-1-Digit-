import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_mlp_model.h5")

st.title("Digit Predictor 🔢")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # grayscale
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.image(uploaded_file, caption="Uploaded Image")
    st.write(f"### Predicted Digit: {digit}")