import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/tomato_efficientnet_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.markdown(
    """
    Upload one or more tomato leaf images to detect diseases using a **deep learning EfficientNet model**.
    """
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# FILE UPLOADER
# =========================
uploaded_files = st.file_uploader(
    "📂 Drag and drop leaf images here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================
# IMAGE PREVIEW
# =========================
images = []
if uploaded_files:
    st.subheader("📸 Uploaded Images")

    cols = st.columns(4)
    for idx, file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

        with cols[idx % 4]:
            st.image(img_rgb, use_column_width=True)

# =========================
# PREDICT BUTTON
# =========================
if uploaded_files:
    st.markdown("---")
    predict_btn = st.button("🔍 Predict Disease")

    if predict_btn:
        st.subheader("🧠 Prediction Results")

        for idx, img_rgb in enumerate(images):
            img_resized = cv2.resize(img_rgb, IMG_SIZE)
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array, verbose=0)[0]
            class_id = np.argmax(preds)
            confidence = preds[class_id] * 100

            st.markdown(
                f"""
                **Image {idx + 1}**
                - 🦠 **Disease:** `{CLASS_NAMES[class_id]}`
                - 📊 **Confidence:** `{confidence:.2f}%`
                """
            )

            st.progress(float(confidence / 100))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("EfficientNet-based Tomato Disease Detection | Research & Academic Use")
