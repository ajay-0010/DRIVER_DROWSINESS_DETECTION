#All Initializers
closed_counter = 0
total_frames = 0
closed_frames = 0
WINDOW_SIZE = 120
PERCLOS_THRESHOLD = 0.40
THRESHOLD = 0.225
perclos = 0
closed_frames_stable = 0
open_frames_stable = 0
STABILITY_FRAMES = 3
status = "Detecting..."

import numpy as np
import time

# Load trained model
from tensorflow.keras.models import load_model
model = load_model("models/eye_model.keras")

# Load Haar face cascade
import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#This is for music.
import pygame
pygame.init()
pygame.mixer.init()


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = gray[y:y+h, x:x+w]

        # ---- Better Eye Region Estimation ----
        eye_top = int(h * 0.32)
        eye_bottom = int(h * 0.50)

        eye_strip = face_roi[eye_top:eye_bottom, :]

        # Slight horizontal margins to avoid side noise
        left_eye = eye_strip[:, int(w*0.05):int(w*0.45)]
        right_eye = eye_strip[:, int(w*0.55):int(w*0.95)]

        eye_status_list = []


        for eye_region in [left_eye, right_eye]:

            try:
                eye_img = cv2.resize(eye_region, (64, 64))
            except:
                continue

            eye_img = eye_img / 255.0
            eye_img = np.reshape(eye_img, (1, 64, 64, 1))

            prediction = model.predict(eye_img, verbose=0)
            eye_status_list.append(prediction[0][0])

        # ---- Decision Logic ----
        if len(eye_status_list) > 0:

            avg_prediction = max(eye_status_list)

            total_frames += 1

            if avg_prediction < THRESHOLD:

                closed_frames_stable += 1
                open_frames_stable = 0

                if closed_frames_stable >= STABILITY_FRAMES:
                    status = "Closed"
                    closed_frames += 1
                    closed_counter += 1

            else:

                open_frames_stable += 1
                closed_frames_stable = 0

                if open_frames_stable >= STABILITY_FRAMES:
                    status = "Open"
                    closed_counter = 0

            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)


        # -------- PERCLOS Calculation --------
        if total_frames >= WINDOW_SIZE:

            perclos = closed_frames / total_frames

            if perclos > PERCLOS_THRESHOLD:
                cv2.putText(frame, "DROWSY (PERCLOS)",
                            (100,150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,0,255), 3)

            total_frames = 0
            closed_frames = 0


        cv2.putText(frame, f"PERCLOS: {perclos:.2f}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)


        # ---- Alert Logic ----
        if closed_counter > 20:

            cv2.putText(frame, "DROWSY ALERT!",
                        (100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 3)

            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load("music/alert.mp3")
                pygame.mixer.music.play()

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()