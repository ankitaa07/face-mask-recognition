# 😷 Face Mask Recognition System

A real-time face mask detection system built using **Deep Learning** and **Computer Vision**.  
The system detects whether a person is wearing a face mask through a live webcam feed and triggers a **sound alert** when a person is detected **without a mask**.

This project demonstrates practical application of CNNs for real-world safety and compliance monitoring.

---

## ✨ Key Features
- Real-time face detection using webcam
- Binary classification: **Mask / No Mask**
- Bounding box with label overlay
- **Sound alert when a no-mask face is detected**
- Fast and lightweight real-time inference

---

## 🧠 Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Imutils  
- H5Py  

---

## 📂 Project Structure
face-mask-recognition/
│── src/
│ ├── detect_mask_video.py
│ └── mask_detector_model.h5
│── requirements.txt
│── .gitignore
│── README.md

---

## ⚙️ Setup Instructions


1️⃣ Clone the repository
bash
git clone https://github.com/ankitaa07/face-mask-recognition.git
cd face-mask-recognition

2️⃣ Create a virtual environment
python -m venv venv

3️⃣ Activate the virtual environment
Windows
venv\Scripts\activate

4️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Application
python src/detect_mask_video.py
- The webcam starts automatically.
- A sound alert is triggered when a person without a mask is detected.
- Press Q to stop the program.

📊 Model Details

-Convolutional Neural Network (CNN)
-Trained for binary classification (Mask / No Mask)
-Model stored in .h5 format
-Optimized for real-time webcam usage

🚀 Future Enhancements

-Display prediction confidence percentage
-Transfer learning using MobileNet
-Email or SMS alerts
-Web deployment using Flask / FastAPI
-Face recognition with identity tracking

👩‍💻 Author

Ankita Mundra
CSE Undergraduate
Interested in Artificial Intelligence, Machine Learning, and Computer Vision
