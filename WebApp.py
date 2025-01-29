import streamlit as st
import cv2
import os
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
import sqlite3
import io
import time
import av

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

# Function to save image and encoding to the database
def save_image_to_db(user_id, image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        encoding = model(image_tensor).cpu().numpy().flatten()
    
    encoding = encoding / np.linalg.norm(encoding)

    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_blob = output.getvalue()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (user_id, image) VALUES (?, ?)', (user_id, image_blob))
    cursor.execute('INSERT INTO encodings (user_id, encoding) VALUES (?, ?)', (user_id, encoding.tobytes()))
    conn.commit()
    conn.close()

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

# Function to delete a user by name
def delete_user_by_name(name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            cursor.execute("DELETE FROM images WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        else:
            conn.close()
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

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
    .stButton button:before {
        content: '';
        background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000);
        position: absolute;
        top: -2px;
        left: -2px;
        background-size: 400%;
        z-index: -1;
        filter: blur(5px);
        width: calc(100% + 4px);
        height: calc(100% + 4px);
        animation: glowing 20s linear infinite;
        opacity: 0;
        transition: opacity .3s ease-in-out;
        border-radius: 10px;
    }
    .stButton button:hover:before {
        opacity: 1;
    }
    @keyframes glowing {
        0% { background-position: 0 0; }
        50% { background-position: 400% 0; }
        100% { background-position: 0 0; }
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
frame_placeholder = st.empty()

# Register User
if register_button:
    if person_name:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name) VALUES (?)', (person_name,))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        directions = ["Look Center", "Look Left", "Look Right", "Look Up", "Look Down"]
        images_per_direction = 12
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)

        captured_frames = []

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            rgb_frame = image[:, :, ::-1]  # Convert BGR to RGB
            
            height, width, _ = image.shape
            box_width, box_height = 200, 200
            center_x, center_y = width // 2, height // 2
            x1, y1 = center_x - box_width // 2, center_y - box_height // 2
            x2, y2 = center_x + box_width // 2, center_y + box_height // 2

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Please {directions[len(captured_frames) // images_per_direction]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if len(captured_frames) < len(directions) * images_per_direction:
                face = image[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).convert('RGB')
                save_image_to_db(user_id, face_pil)
                captured_frames.append(face)
                progress_bar.progress(len(captured_frames) / (len(directions) * images_per_direction))
            else:
                st.success(f"Successfully captured images of {person_name} from different angles.")
                webrtc_streamer.stop()
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(key="face-registration", mode=WebRtcMode.SENDRECV, video_frame_callback=video_frame_callback)

# Delete User
if delete_button:
    if delete_name:
        if delete_user_by_name(delete_name):
            st.success(f"User '{delete_name}' and their associated images have been deleted successfully!")
        else:
            st.error(f"User '{delete_name}' not found in the database.")
    else:
        st.error("Please enter a valid name.")

# Start Face Recognition
encodings = fetch_encodings_from_db()

start_button = st.button("Start Face Recognition")
stop_button = st.button("Stop Face Recognition")
frame_placeholder = st.empty()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    img_rgb = image[:, :, ::-1]
    boxes, _ = mtcnn.detect(img_rgb)
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            face_tensor = preprocess_face(img_rgb, box)
            with torch.no_grad():
                encoding = model(face_tensor).cpu().numpy().flatten()
                encoding = encoding / np.linalg.norm(encoding)
            
            best_match, best_score = "Unknown", 0.5
            for person, person_encoding in encodings.items():
                score = np.dot(encoding, person_encoding)
                if score > best_score:
                    best_match = person
                    best_score = score
            
            label = f"{best_match} ({round(best_score * 100, 2)}%)"
            color = (0, 255, 0) if best_match != "Unknown" else (255, 0, 0)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    else:
        cv2.putText(image, "Face Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")

if start_button:
    webrtc_streamer(key="face-recognition", mode=WebRtcMode.SENDRECV, video_frame_callback=video_frame_callback)

if stop_button:
    webrtc_streamer.stop()
