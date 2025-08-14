# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import glob
import tempfile

st.set_page_config(page_title="YOLOv11 Object Detector", page_icon="📦")
st.title("📦 Object Detection using YOLOv11")
st.write("Upload an image or use your webcam for real-time object detection 🪄")

# Load YOLO model once
model = YOLO("yolo11s.pt")  # Auto-downloads if not present

# Option: Upload image OR use camera
mode = st.radio("Choose Detection Mode:", ["📤 Upload Image", "📷 Live Camera"])

if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save image
        image_path = os.path.join("input.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(image_path, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("Detecting objects... 🚀"):
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

elif mode == "📷 Live Camera":
    st.warning("Press 'Start' to begin detection. Press 'Stop' to end.")

    start_button = st.button("▶ Start Camera")
    stop_button = st.button("⏹ Stop Camera")

    if start_button:
        cap = cv2.VideoCapture(0)  # 0 = default webcam
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame)
            annotated_frame = results[0].plot()

            # Show frame
            stframe.image(annotated_frame, channels="BGR")

            if stop_button:
                break

        cap.release()
        st.success("Camera stopped.")
