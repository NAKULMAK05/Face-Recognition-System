import streamlit as st
import cv2
import torch
import numpy as np
import sqlite3
import io
import av
from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from PIL import Image
from torchvision import transforms

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

db_path = 'face_recognition.db'

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encodings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        encoding BLOB NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

init_db()

def save_encoding_to_db(user_id, encoding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO encodings (user_id, encoding) VALUES (?, ?)', (user_id, encoding.tobytes()))
    conn.commit()
    conn.close()

def fetch_encodings():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT users.name, encodings.encoding FROM users JOIN encodings ON users.id = encodings.user_id')
    data = cursor.fetchall()
    conn.close()
    encodings = {}
    for name, encoding_blob in data:
        encoding = np.frombuffer(encoding_blob, dtype=np.float32)
        encodings[name] = encoding
    return encodings

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.encodings = fetch_encodings()
    
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img_rgb[y1:y2, x1:x2]
                face_pil = Image.fromarray(face).convert('RGB')
                face_tensor = transforms.ToTensor()(face_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    encoding = model(face_tensor).cpu().numpy().flatten()
                    encoding = encoding / np.linalg.norm(encoding)
                best_match, best_score = "Unknown", 0.5
                for name, stored_encoding in self.encodings.items():
                    score = np.dot(encoding, stored_encoding)
                    if score > best_score:
                        best_match, best_score = name, score
                label = f"{best_match} ({round(best_score * 100, 2)}%)"
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return img

st.title("Real-Time Face Recognition with WebRTC")
st.markdown("""
    <style>
        .stButton button {
            width: 220px; height: 50px; border: none; background: #111;
            color: #fff; font-size: 16px; border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

name = st.text_input("Enter User's Name")
register_button = st.button("Register User")

if register_button and name:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    st.success(f"User {name} registered successfully!")

delete_name = st.text_input("Enter Name to Delete:")
delete_button = st.button("Delete User")

if delete_button and delete_name:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE name = ?", (delete_name,))
    user = cursor.fetchone()
    if user:
        user_id = user[0]
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        cursor.execute("DELETE FROM encodings WHERE user_id = ?", (user_id,))
        conn.commit()
        st.success(f"Deleted user {delete_name}.")
    else:
        st.error("User not found.")
    conn.close()

webrtc_streamer(key="face-recognition", video_processor_factory=VideoProcessor, mode=WebRtcMode.RECVONLY)
