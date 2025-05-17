# 🔐 InSight ID – AI-Based Face Recognition

An advanced and accurate real-time face recognition application built with Python, Streamlit, and a pre-trained FaceNet model. This system enables users to register and recognize faces using webcam input, offering a user-friendly interface with glowing buttons for enhanced user experience.

---

## 🚀 Features

* **Face Registration**: Capture images from multiple angles to register a user in the database.
* **Real-Time Recognition**: Recognize multiple faces in live webcam feed.
* **User Management**: Add and manage user profiles.
* **Aesthetic UI**: Interactive interface with glowing buttons for better user experience.

---

## 🛠️ Technologies Used

* **Programming Language**: Python 3.8+
* **Libraries**:

  * Streamlit
  * OpenCV
  * FaceNet-PyTorch
  * Torch
  * Pillow
  * NumPy
  * SQLite3 (built-in with Python)
  * Torchvision

---

## 📁 Repository Structure

```

├── Face Recognition (1).ipynb    # Jupyter notebook for model development
├── Final_Face_Recog.py           # Core face recognition script
├── WebApp.py                     # Main Streamlit application
├── requirements.txt              # List of dependencies
├── packages.txt                  # Additional package information
└── README.md                     # Project documentation
```



---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/NAKULMAK05/Face-Recognition-System.git
   cd Face-Recognition-System
   ```



2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install the required Python libraries**:

   ```bash
   pip install -r requirements.txt
   ```



4. **Verify GPU support for better performance (optional)**:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```



---

## 🖥️ How to Run the Application

1. **Start the Streamlit app**:

   ```bash
   streamlit run WebApp.py
   ```



2. **Open the URL displayed in your terminal (e.g., [http://localhost:8501](http://localhost:8501)) in your web browser**.

---

## 🧪 Usage Instructions

### Register a User

* Enter the name of the user in the input box.
* Click "Register User" and follow the on-screen directions (e.g., "Look Left").
* Once completed, the user will be added to the database.

### Start Face Recognition

* Click "Start Face Recognition".
* The app will display a live webcam feed, highlighting recognized faces and displaying their names.

### Stop Face Recognition

* Click "Stop Face Recognition" to end the recognition process.

---

## 📷 Outputs

* **Home Page**: Landing page of the application.
* **Face Registration**: Interface for registering new users.
* **Face Recognition**: Real-time face recognition interface.
* **Multi-Face Recognition**: Recognize multiple faces simultaneously.
* **Edge Cases**: Handles unknown faces or when a face is not detected.

---

## 🔧 Troubleshooting

* **Webcam Not Working**:

  * Ensure your webcam is properly connected.
  * Restart the app and check permissions for webcam access.

* **Face Not Detected**:

  * Ensure the lighting is sufficient.
  * Adjust your position for better visibility of your face.

* **Performance Issues**:

  * Check GPU support for faster processing.
  * Reduce the number of faces to be recognized simultaneously.

---

## 📈 Future Scope

* Integration into automated attendance systems.
* Deployment for surveillance and security applications.
* Enhancements for mobile and edge devices.

---

**Outputs**


<br/>
<br/>

  
**1)Home Page**


  <br/>
  <br/>
  <img width="959" alt="facerec" src="https://github.com/user-attachments/assets/511d5031-ed4f-4e66-8de7-87e4e1289bb4" />
  <br/>
  <br/>
  
  **2)Face Registration**

  
  <br/>
  <br/>
  <img width="959" alt="reg" src="https://github.com/user-attachments/assets/ddb6c103-4790-44da-81bf-616fcfe4e3e2" />
     <br/>
     <br/>

     
 **3)Face Recognition**
<br/>
<br/>
 <img width="959" alt="recog" src="https://github.com/user-attachments/assets/9099cbbc-b1b1-4eef-bd19-814934fd8901" />

  <br/>
  <br/>

 **4)Multi Faces Recognition**
<br/>
<br/>
<img width="959" alt="MultiFaces" src="https://github.com/user-attachments/assets/fce3e652-d214-4aa7-bf4b-2fb5992de398" />
  <br/>
  <br/>

  **5)Edge Cases (Unknown or Face Not Detected)**
  <br/>
  <br/>
<img width="959" alt="unk" src="https://github.com/user-attachments/assets/426bad9e-5858-4001-bfc4-6d58bea805ba" />
<img width="959" alt="FND" src="https://github.com/user-attachments/assets/25cf41ed-c10a-4748-84b6-a9923bce3fc8" />

<br/><br/>

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Empowering secure and seamless identification through cutting-edge technology.*

---
