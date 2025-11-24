📦 YOLOv11 Object Detection App

Real-time & image-based object detection using Streamlit + YOLOv11.

🚀 Overview

This project is a lightweight and fast object detection web app built using Streamlit and Ultralytics YOLOv11.
You can:

📤 Upload images for detection

📷 Use your webcam for real-time detection

🔍 View results with bounding boxes

⚡ Enjoy a clean & smooth UI

🧠 Features

YOLOv11s model for fast inference

Real-time camera detection using streamlit-webrtc

Auto-generated results saved in runs/detect/

Works on CPU (no GPU required)

Minimal & user-friendly interface

🛠️ Tech Stack

Python

Streamlit

Ultralytics YOLOv11

OpenCV

Pillow

streamlit-webrtc

📂 Project Structure
├── app.py
├── requirements.txt
├── README.md
└── runs/
    └── detect/
        └── predict*/   # auto-generated output images

🔧 Installation
1️⃣ Clone the repo
git clone https://github.com/your-username/yolo11-object-detection.git
cd yolo11-object-detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

📝 Requirements
streamlit
ultralytics
Pillow
opencv-python-headless
streamlit-webrtc
av

📘 How It Works

Loads the YOLOv11 model once

Detects objects in uploaded images

Detects objects in real-time via webcam

Annotated frames/images are displayed directly in the app

🎯 Future Improvements

Add FPS counter

Download detected images

Multi-object tracking (ByteTrack/DeepSORT)

Better UI/UX animations

🤝 Contributing

Contributions, issues, and feature requests are welcome.

⭐ Support

If you like this project, please star ⭐ the repository.
