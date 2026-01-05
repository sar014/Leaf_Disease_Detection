import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from leaf_disease_detection import detect_disease

@st.cache_data
def load_dataset():
    return pd.read_csv("leaf_disease_severity_with_env.csv")

df = load_dataset()
available_diseases = sorted(df["disease"].unique())

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_segmentation(image_path):
    try:
        percentage, disease_mask = detect_disease(image_path)
        return percentage
    except Exception as e:
        st.error(f"Segmentation failed: {e}")
        return None


st.set_page_config(layout="wide")
st.title("Leaf Disease Segmentation & Severity Analysis")

st.markdown("""
This application segments diseased regions on leaf images and estimates disease severity.
Environmental correlation is shown only for diseases present in the reference dataset.
""")


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png",".jfif"]
)

# ---------- K Selector ----------
k = st.slider(
    "Select number of KMeans clusters (K)",
    min_value=2,
    max_value=100,
    value=4,
    step=1
)

selected_disease = st.selectbox(
    "Select disease present in the leaf",
    options=["-- Select --"] + available_diseases
)


if uploaded_file is not None and selected_disease != "-- Select --":

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.subheader("Segmentation Results")

    with st.spinner("Running disease segmentation..."):
        output = detect_disease(image_path,k)

    st.success(f"Disease Severity: **{output['percentage']:.2f}%**")

    if output is None:
        st.error("Failed to process image.")
    else:
        st.success(f"Disease Severity: **{output['percentage']:.2f}%**")

        st.subheader("Segmentation Pipeline Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(to_rgb(output["original"]), caption="Original Image")
            st.image(to_rgb(output["enhanced"]), caption="CLAHE Enhanced")

        with col2:
            st.image(output["leaf_mask"], caption="Leaf Mask", clamp=True)
            st.image(output["cluster_img"], caption="K-Means Clusters", clamp=True)

        with col3:
            st.image(output["disease_mask"], caption="Disease Mask", clamp=True)
            st.image(to_rgb(output["segmented"]), caption="Segmented Disease Regions")

        # Overlay
        overlay = output["original"].copy()
        overlay[output["disease_mask"] > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(output["original"], 0.7, overlay, 0.3, 0)

        st.image(to_rgb(overlay), caption="Disease Overlay")

    if selected_disease in available_diseases:

        st.subheader("Environmental Correlation Analysis")

        disease_df = df[df["disease"] == selected_disease]

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.scatter(
                disease_df["humidity_pct"],
                disease_df["disease_percentage"],
                alpha=0.6
            )
            ax1.set_xlabel("Humidity (%)")
            ax1.set_ylabel("Disease Severity (%)")
            ax1.set_title("Disease Severity vs Humidity")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.scatter(
                disease_df["temperature_c"],
                disease_df["disease_percentage"],
                alpha=0.6
            )
            ax2.set_xlabel("Temperature (°C)")
            ax2.set_ylabel("Disease Severity (%)")
            ax2.set_title("Disease Severity vs Temperature")
            st.pyplot(fig2)

    else:
        st.warning("Environmental correlation not available for this disease.")

    os.remove(image_path)


