import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Local source imports
from src.models import FolivusNet

# 1. Page Configuration & Brand Styling Setup
st.set_page_config(
    page_title="Folivus AI | Crop Diagnostics", page_icon="🌿", layout="centered"
)

# Inject custom CSS to apply the official Folivus Brand Palette
st.markdown(
    """
    <style>
        .main { background-color: #ECEFF1; }
        h1 { color: #37474F; font-family: 'Montserrat', sans-serif; font-weight: 700; }
        h3 { color: #2E7D32; }
        .stButton>button {
            background-color: #2E7D32;
            color: white;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover { background-color: #A5D6A7; color: #37474F; }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Helper Functions for Loading Configs & Cached Models
@st.cache_resource
def load_environment():
    """Loads YAML configurations safely."""
    with open(os.path.join("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_folivus_model(config):
    """Instantiates model architecture and loads saved weights safely."""
    device = torch.device("cpu")  # Force CPU execution for web hosting stability
    model = FolivusNet(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    )
    weights_path = config["training"]["model_save_path"]
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model
    else:
        st.error(
            f"Could not locate trained weight map at '{weights_path}'! Run training operations first."
        )
        return None


def preprocess_image(pil_image, img_size):
    """
    Applies our exact calibrated empirical leaf normalizations
    derived during our EDA audit phase.
    """
    # Convert PIL to openCV matrix format (RGB)
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    # Apply our exact Folivus empirical channel metrics
    mean = np.array([0.456, 0.478, 0.409], dtype=np.float32)
    std = np.array([0.185, 0.157, 0.198], dtype=np.float32)
    img = (img - mean) / std

    # Rearrange dimensions from HWC to PyTorch Standard BCHW format
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)


# 3. App UI Layout Header
st.title("🌿 Project Folivus")
st.subheader("Production-Grade Crop Disease Diagnostics Engine")
st.markdown(
    "Upload a close-up photograph of a plant leaf to analyze it for target pathologies using our custom trained neural network."
)
st.write("---")

# Initialize setup backend
config = load_environment()
model = load_folivus_model(config)

# Static listing of classes ordered alphabetically to match dataset parsing
CLASS_NAMES = [
    "Pepper Bell: Bacterial Spot",
    "Pepper Bell: Healthy",
    "Potato: Early Blight",
    "Potato: Healthy",
    "Potato: Late Blight",
    "Tomato: Bacterial Spot",
    "Tomato: Early Blight",
    "Tomato: Healthy",
    "Tomato: Late Blight",
    "Tomato: Leaf Mold",
    "Tomato: Septoria Leaf Spot",
    "Tomato: Spider Mites (Two-spotted)",
    "Tomato: Target Spot",
    "Tomato: Yellow Leaf Curl Virus",
    "Tomato: Mosaic Virus",
]

# 4. Image Upload & Interface Interactive Pipeline Execution Area
uploaded_file = st.file_uploader(
    "Choose a leaf photograph...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)

    st.write("")
    if st.button("🚀 Run Folivus Diagnostics Engine"):
        if model is not None:
            with st.spinner("Analyzing cellular leaf patterns..."):
                # Run preprocessing and inference
                tensor_img = preprocess_image(image, config["data"]["img_size"])
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, class_idx = torch.max(probabilities, 0)

                # Render results dynamically
                predicted_class = CLASS_NAMES[class_idx.item()]
                confidence_score = confidence.item() * 100

                st.success("### Analysis Complete!")

                # Check pathology state to color code status boxes safely
                if "healthy" in predicted_class.lower():
                    st.balloons()
                    st.info(
                        f"**Diagnosis:** {predicted_class} (Confidence: {confidence_score:.2f}%)"
                    )
                else:
                    st.warning(
                        f"**Pathology Detected:** {predicted_class} (Confidence: {confidence_score:.2f}%)"
                    )

                # Render probability distribution tracking bars
                st.write("#### Confidence Breakdown per Category:")
                for idx, name in enumerate(CLASS_NAMES):
                    prob = probabilities[idx].item()
                    if prob > 0.01:  # Only display classes with >1% prediction trace
                        st.progress(prob, text=f"{name}: {prob*100:.1f}%")
