# Face Recognition System
The image recognition project aims to develop a system capable of recognising individuals using a laptop's camera. This system leverages computer vision techniques to capture real-time images and process them for face identification.<br/>

**This is a real-time face recognition app built with Python, Streamlit, and FaceNet PreTrained model . It enables users to register and recognize faces using webcam input.**

**Features**<br/>
* **Face Registration**: Capture images from multiple angles to register a user in the database.<br/>
* **Real-Time Recognition**: Recognize faces in live webcam feed.<br/>  
* **User Management**: Add user profiles.<br/>    
* **Glowing Button UI**: Aesthetic UI design with glowing buttons for better user experience.<br/> 
<br/>
     
**Face Recogntion.ipynb** : This file contains the Python code for the Face Recognition Model<br/>  
**WebApp.py** : This file is the main application for Face Recognition . Run This File  
using the instructions given below. <br/>
 
**Requirements**<br/> 
Ensure you have the following installed on your system:<br/>  
 
* Python 3.8 or higher<br/> 
* A webcam<br/> 
* Libraries mentioned in the requirements.txt<br/> 
<br/>
    
**Installation** 
<br/> 
* **Clone this repository**:<br/>
https://github.com/NAKULMAK05/Face-Recognition-System.git <br/> 
<br/>  
  
* **Install the required Python libraries**:<br/> 
pip install -r requirements.txt 
 
**Required Libraries:**<br/> 

* streamlit
* opencv-python
* facenet-pytorch
* torch 
* Pillow
* numpy
* sqlite3 (builtin with Python) 
* torchvision

<br/>

**Verify GPU support for better performance (optional):** 
<br/>
python -c "import torch; print(torch.cuda.is_available())" <br/>
<br/>
If False, the app will run on CPU.
<br/>
<br/>
**How to Run the App**<br/>
<br/>
* Start the app: streamlit run FaceRecognitionApp.py <br/>
* Open the URL displayed in your terminal (e.g., http://localhost:8501) in your web browser.<br/>
<br/>
<br/>

**Usage Instructions**
<br/>
<br/>

**Register a User**<br/>
<br/>
* Enter the name of the user in the input box.<br/>
* Click "Register User" and follow the on-screen directions (e.g., "Look Left").<br/>
* Once completed, the user will be added to the database.<br/>
<br/>

**Start Face Recognition**<br/>
<br/>
* Click "Start Face Recognition".<br/>
* The app will display a live webcam feed, highlighting recognized faces and displaying their names.<br/>

<br/>
<br/>

**Stop Face Recognition**
<br/>
<br/>
* Click "Stop Face Recognition" to end the recognition process.<br/>
<br/>
<br/>

**Troubleshooting**
<br/>
<br/>
**Webcam Not Working**
* Ensure your webcam is properly connected.<br/>
* Restart the app and check permissions for webcam access.<br/>

<br/>

**Face Not Detected**
* Ensure the lighting is sufficient.<br/>
* Adjust your position for better visibility of your face.<br/>
  <br/>

**Performance Issues**
* Check GPU support for faster processing.<br/>
* Reduce the number of faces to be recognized simultaneously.<br/>
<br/>
<br/>


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

