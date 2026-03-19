'''import streamlit as st
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
    st.write(f"### Predicted Digit: {digit}")'''


import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from streamlit_drawable_canvas import st_canvas



# Load model
session = ort.InferenceSession("mnist_mlp_model.onnx")
input_name = session.get_inputs()[0].name

st.title("Digit Predictor 🔢")

tab1, tab2 = st.tabs(["✏️ Draw a Digit", "📁 Upload an Image"])

# ─── TAB 1: Draw ───────────────────────────────────────────
with tab1:
    st.write("Draw a digit below and click **Predict**")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from Drawing"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            img = img.convert("L")              # grayscale
            img = img.resize((28, 28))
            img = np.array(img) / 255.0
            img = img.reshape(1, 784).astype(np.float32)

            result = session.run(None, {input_name: img})
            digit = np.argmax(result[0])
            confidence = float(np.max(result[0]))

            st.success(f"### Predicted Digit: {digit}")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.warning("Please draw a digit first!")

# ─── TAB 2: Upload ─────────────────────────────────────────
with tab2:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 784).astype(np.float32)

        result = session.run(None, {input_name: img})
        digit = np.argmax(result[0])
        confidence = float(np.max(result[0]))

        st.image(uploaded_file, caption="Uploaded Image", width=200)
        st.success(f"### Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2%}")

'''### Update `requirements.txt` to:
```
onnxruntime
streamlit
numpy
Pillow
streamlit-drawable-canvas'''