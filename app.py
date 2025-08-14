import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import glob
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="YOLOv11 Object Detector", page_icon="📦")
st.title("📦 Object Detection using YOLOv11")
st.write("Upload an image or use your webcam for real-time object detection 🪄")

# Load YOLO model once
model = YOLO("yolo11s.pt")  # Auto-downloads if not present

# Option: Upload image OR use camera
mode = st.radio("Choose Detection Mode:", ["📤 Upload Image", "📷 Live Camera"])

# -------------------
# Upload Image Section
# -------------------
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

# -------------------
# Live Camera Section
# -------------------
elif mode == "📷 Live Camera":
    st.info("Click 'Start' to activate your webcam for detection.")

    class YOLODetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated_img = results[0].plot()
            return annotated_img

    webrtc_streamer(
        key="yolo-live",
        video_transformer_factory=YOLODetector,
        media_stream_constraints={"video": True, "audio": False}
    )
