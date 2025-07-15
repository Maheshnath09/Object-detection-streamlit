# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import glob

st.set_page_config(page_title="YOLOv11 Object Detector", page_icon="📦")
st.title("📦 Object Detection using YOLOv11")
st.write("Upload an image and let the model detect objects like magic 🪄")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save image
    image_path = os.path.join("input.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects... 🚀"):
        # Load YOLO model
        model = YOLO("yolo11s.pt")  # Auto-downloads if not present

        # Run prediction
        results = model.predict(source=image_path, conf=0.3, save=True)

        # Auto-find result folder & file
        output_dirs = sorted(glob.glob("runs/detect/predict*"), key=os.path.getmtime, reverse=True)
        if output_dirs:
            result_folder = output_dirs[0]
            image_files = glob.glob(f"{result_folder}/*.jpg")
            if image_files:
                st.success("✅ Detection complete!")
                st.image(image_files[0], caption="🔍 Detected Image", use_column_width=True)
            else:
                st.warning("⚠️ No output image found.")
        else:
            st.warning("⚠️ Prediction folder not found.")
