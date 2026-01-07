import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from disease_pipeline import detect_disease
from correlation import show_environment_correlation

@st.cache_data
def load_dataset():
    return pd.read_csv("leaf_disease_severity_with_env.csv")

df = load_dataset()
available_diseases = sorted(df["disease"].unique())

def colorize_clusters(label_img):
    label_img = label_img.astype(np.uint8)

    # Normalize to 0–255
    norm = cv2.normalize(label_img, None, 0, 255, cv2.NORM_MINMAX)

    # Apply colormap
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

st.set_page_config(layout="wide")
st.title("🌿 Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg","jfif"]
)

k = st.slider(
    "Select number of clusters (K)",
    min_value=2,
    max_value=20,
    value=4,
    step=1
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Processing image..."):
        img, leaf_mask, disease_mask, seg_img, clusters, percentage = detect_disease(
            img_bgr, k
        )

    col1, col2, col3 = st.columns(3)

    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Enhanced Image")
    col2.image(leaf_mask, caption="Leaf Mask", clamp=True)
    cluster_viz = colorize_clusters(clusters)
    col3.image(cluster_viz, caption=f"KMeans Clusters (K={k})")


    col4, col5, col6 = st.columns(3)
    col4.image(disease_mask, caption="Disease Mask", clamp=True)
    col5.image(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB), caption="Segmented Disease")

    overlay = img.copy()
    overlay[disease_mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    col6.image(
        cv2.cvtColor(blended, cv2.COLOR_BGR2RGB),
        caption=f"Disease Overlay ({percentage:.2f}%)"
    )

    st.success(f"🌡 Disease Percentage: {percentage:.2f}%")

    
    selected_disease = st.selectbox(
        "Select disease present in the leaf",
        options=["-- Select --"] + available_diseases
    )

    show_environment_correlation(df, selected_disease)
