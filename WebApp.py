import streamlit as st
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
import sqlite3 
import io
import time

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
    
    # Normalize the embedding
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

# Custom CSS for Aesthetic UI
st.markdown("""
    <style>
    /* Full-page styling */
    .stApp {
        background: linear-gradient(135deg, #1a0d2e 0%, #0d1b2a 100%);
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
    }
    
    /* Title styling */
    h1 {
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.7);
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #2a2a3d;
        border: 2px solid #00ffcc;
        border-radius: 8px;
        color: #e0e0e0;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        max-width: 300px;
    }
    .stTextInput > div > div > input:focus {
        outline: none;
        box-shadow: 0 0 5px #00ffcc;
    }
    .stTextInput label {
        color: #00ffcc;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff00cc, #00ffcc);
        border: none;
        border-radius: 12px;
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        padding: 12px 24px;
        margin: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.8);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(to right, #00ffcc, #ff00cc);
        border-radius: 5px;
        height: 10px;
    }
    
    /* Container for content */
    .content-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        max-width: 600px;
        width: 100%;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.2);
    }
    
    /* Success and error messages */
    .stSuccess {
        background-color: rgba(0, 255, 204, 0.1);
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 5px;
    }
    .stError {
        background-color: rgba(255, 0, 102, 0.1);
        color: #ff0066;
        border: 1px solid #ff0066;
        border-radius: 5px;
    }
    
    /* Animation for glowing effect */
    @keyframes glow {
        0% { box-shadow: 0 0 5px #00ffcc; }
        50% { box-shadow: 0 0 20px #00ffcc; }
        100% { box-shadow: 0 0 5px #00ffcc; }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎭 Real-Time Face Recognition System")

# Center content in a container
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Input and Button Section
    col1, col2 = st.columns(2)
    with col1:
        person_name = st.text_input("Enter User's Name", placeholder="e.g., John Doe")
        register_button = st.button("Register User")
    
    with col2:
        delete_name = st.text_input("Delete User by Name", placeholder="e.g., John Doe")
        delete_button = st.button("Delete User")

    # Face Recognition Controls
    col3, col4 = st.columns(2)
    with col3:
        start_button = st.button("Start Face Recognition")
    with col4:
        stop_button = st.button("Stop Face Recognition")

    st.markdown('</div>', unsafe_allow_html=True)

# Frame placeholder for camera
frame_placeholder = st.empty()

# ---------------------------------------------------------
# 1) REGISTER USER
# ---------------------------------------------------------
if register_button:
    if person_name:
        # Insert user into DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name) VALUES (?)', (person_name,))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Open webcam using DirectShow
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            # Set known resolution & FourCC to avoid glitchy frames
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Warm-up frames
            for _ in range(5):
                cap.read()
            
            directions = ["Look Center", "Look Left", "Look Right", "Look Up", "Look Down"]
            count = 0
            images_per_direction = 12
            progress_bar = st.progress(0)

            # Dimensions for the central bounding box
            box_width, box_height = 200, 200
            center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
            center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

            for direction in directions:
                time.sleep(1)  # Give user time to move
                for _ in range(images_per_direction):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error: Could not read from webcam.")
                        break
                    
                    x1 = center_x - box_width // 2
                    y1 = center_y - box_height // 2
                    x2 = center_x + box_width // 2
                    y2 = center_y + box_height // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Please {direction}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Crop face within the bounding box
                    face = frame[y1:y2, x1:x2]
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).convert('RGB')
                    save_image_to_db(user_id, face_pil)

                    count += 1
                    progress_bar.progress(count / (len(directions) * images_per_direction))
                    frame_placeholder.image(frame, channels="BGR", clamp=True)
                    time.sleep(0.1)
            
            cap.release()
            st.success(f"Successfully captured images of {person_name} from different angles.")

# ---------------------------------------------------------
# 2) DELETE USER
# ---------------------------------------------------------
if delete_button:
    if delete_name:
        if delete_user_by_name(delete_name):
            st.success(f"User '{delete_name}' and their associated images have been deleted successfully!")
        else:
            st.error(f"User '{delete_name}' not found in the database.")
    else:
        st.error("Please enter a valid name.")

# ---------------------------------------------------------
# 3) START FACE RECOGNITION
# ---------------------------------------------------------
if start_button:
    # Load encodings from the database
    encodings = fetch_encodings_from_db()
    
    # Open webcam with DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        # Set resolution & FourCC
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm-up frames
        for _ in range(5):
            cap.read()

        st.write("Face recognition started. Press 'Stop' to end.")
        
        while True:
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            # Convert to NumPy and add batch dimension
            img_np = np.array(img_pil)
            imgs_batch = np.expand_dims(img_np, axis=0)

            boxes_batch, _ = mtcnn.detect(imgs_batch)
            boxes = boxes_batch[0] if boxes_batch is not None else None

            if boxes is not None and len(boxes) > 0:
                img_rgb = np.array(img_pil)  # RGB array
                for box in boxes:
                    face_tensor = preprocess_face(img_rgb, box)
                    with torch.no_grad():
                        encoding = model(face_tensor).cpu().numpy().flatten()
                        encoding = encoding / np.linalg.norm(encoding)
                    
                    # Find best match
                    best_match, best_score = "Unknown", 0.5
                    for person, person_encoding in encodings.items():
                        score = np.dot(encoding, person_encoding)
                        if score > best_score:
                            best_match = person
                            best_score = score
                    
                    label = f"{best_match} ({round(best_score * 100, 2)}%)"
                    color = (0, 255, 0) if best_match != "Unknown" else (255, 0, 0)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            else:
                cv2.putText(frame, "Face Not Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Display the frame in Streamlit
            frame_placeholder.image(frame, channels="BGR", clamp=True)

        cap.release()
        st.write("Face recognition stopped.")
