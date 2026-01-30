import cv2
import numpy as np
from keras.models import load_model
import winsound  # For beep alert
import datetime
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "mask_detector_model.h5")

model = load_model(model_path)

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

# Create alerts directory
os.makedirs("alerts", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        (mask, no_mask) = model.predict(face)[0]

        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Alert system for no-mask detection
        if label == "No Mask":
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            print(f"[ALERT] No Mask detected at {timestamp}")

            # Beep sound
            winsound.Beep(1000, 500)  # 1000 Hz, 500 ms

            # Save the frame with bounding box
            alert_image = frame.copy()
            cv2.rectangle(alert_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(alert_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imwrite(f"alerts/no_mask_{timestamp}.png", alert_image)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
