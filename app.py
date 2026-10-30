import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="AI vs Real Image Detector",
    layout="centered"
)

# Custom minimal CSS for clean typography and cards
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        font-size: 1rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI vs Real Image Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Deep learning transfer learning pipeline powered by InceptionResNetV2 to classify synthetic versus authentic imagery.</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_detector_model():
    local_path = os.path.join("Model", "att_model.keras")
    if os.path.exists(local_path):
        return tf.keras.models.load_model(local_path)
    
    with st.spinner("Downloading model weights from Hugging Face repository..."):
        model_path = hf_hub_download(
            repo_id="bhumika49/ai-vs-real-detector",
            filename="att_model.keras"
        )
        return tf.keras.models.load_model(model_path)

model = load_detector_model()

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing: resize to 224x224 and normalize [0, 1]
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with col2:
        with st.spinner("Extracting spatial features..."):
            prediction = float(model.predict(img_batch)[0][0])
        
        st.subheader("Classification Result")
        
        # AiArtData = class 1, RealArt = class 0
        if prediction > 0.5:
            confidence = prediction * 100
            st.error("**Category:** AI-Generated Artwork")
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
            st.progress(prediction)
        else:
            confidence = (1.0 - prediction) * 100
            st.success("**Category:** Real Image")
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
            st.progress(1.0 - prediction)

        st.caption("Architecture: InceptionResNetV2 | Input Dimensions: 224x224x3")