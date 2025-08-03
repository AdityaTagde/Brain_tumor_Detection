import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# ✅ Set background color only (no text color changes)
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main {
            background-color: #f5f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Default title styling (text color not changed)
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect the type of brain tumor.")

# ✅ Load model
model = load_model("brain_tumor.h5")
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ✅ Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, width=400)

    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(f"🧠{predicted_class.upper()}")