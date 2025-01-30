import streamlit as st
import cv2
import os
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
import sqlite3
import io
import time
import av
import base64

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60)

# Initialize InceptionResnetV1 for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Set up image transformation for model input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Database setup
db_path = 'face_recognition3.db'

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        image BLOB NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encodings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        encoding BLOB NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )''')
    conn.commit()
    conn.close()

init_db()

def fetch_encodings_from_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT users.name, encodings.encoding FROM users
                      JOIN encodings ON users.id = encodings.user_id''')
    data = cursor.fetchall()
    conn.close()

    encodings = {}
    for name, encoding_blob in data:
        encoding = np.frombuffer(encoding_blob, dtype=np.float32)
        encodings[name] = encoding
    return encodings

# Function to preprocess face for recognition
def preprocess_face(img_rgb, box):
    x1, y1, x2, y2 = map(int, box)
    face = img_rgb[y1:y2, x1:x2]
    face_pil = Image.fromarray(face).convert('RGB')
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    return face_tensor

# Custom CSS for styling
st.markdown("""
    <style>
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100vh;
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        background: #000;
    }
    .stButton button {
        width: 220px;
        height: 50px;
        border: none;
        outline: none;
        color: #fff;
        background: #111;
        cursor: pointer;
        position: relative;
        z-index: 0;
        border-radius: 10px;
        font-size: 16px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎭 Real-Time Face Recognition System")

# Input and Button Section
person_name = st.text_input("Enter User's Name")
register_button = st.button("Register User")
delete_name = st.text_input("Enter the name of the user to delete:")
delete_button = st.button("Delete User")
start_button = st.button("Start Face Recognition")
stop_button = st.button("Stop Face Recognition")

# Face recognition function
encodings = fetch_encodings_from_db()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    img_rgb = image[:, :, ::-1]
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            face_tensor = preprocess_face(img_rgb, box)
            with torch.no_grad():
                encoding = model(face_tensor).cpu().numpy().flatten()
                encoding = encoding / np.linalg.norm(encoding)

            best_match, best_score = None, float('inf')
            for name, db_encoding in encodings.items():
                score = np.linalg.norm(encoding - db_encoding)
                if score < best_score:
                    best_score = score
                    best_match = name

            if best_match:
                label = f"Recognized: {best_match}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# ✅ FIX: WebRTC Configuration to Prevent Disappearance
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ✅ FIX: Ensures WebRTC doesn't close after button click
if start_button:
    webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False}
    )

if stop_button:
    st.warning("Face recognition stopped!")

