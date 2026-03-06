# Driver Drowsiness Detection System 🚗😴

A Machine Learning based system that detects driver drowsiness in real-time using a Convolutional Neural Network (CNN) and webcam input. The system monitors the driver's eye state and triggers an alarm if signs of drowsiness are detected.

---

## 📌 Project Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project aims to reduce such accidents by detecting when a driver becomes drowsy and alerting them immediately.

The system uses a trained **CNN model** to classify eye states as **open** or **closed** and calculates **PERCLOS (Percentage of Eye Closure)** to determine if the driver is drowsy.

---

## 🚀 Features

* Real-time webcam monitoring
* Eye state classification (Open / Closed)
* CNN-based deep learning model
* Drowsiness detection using **PERCLOS metric**
* Audio alarm when driver appears drowsy
* Real-time status display
* High accuracy model (**95% accuracy**)

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pygame (for alarm sound)

---

## 📂 Project Structure

```
Driver-Drowsiness-Detection
│
├── model/
│   └── drowsiness_model.h5
│
├── dataset/
│   ├── open_eyes
│   └── closed_eyes
│
├── realtime_drowsiness.py
├── train_model.py
├── evaluate.py
├── alarm.wav
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ajay-0010/DRIVER_DROWSINESS_DETECTION.git
cd DRIVER_DROWSINESS_DETECTION
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Start real-time detection:

```bash
python realtime_drowsiness.py
```

The webcam will open and start monitoring eye states.

If the driver’s eyes remain closed for a certain duration, an **alarm sound will trigger**.

---

## 📊 Model Details

* Model Type: **Convolutional Neural Network (CNN)**
* Input: Eye images
* Output Classes:

  * Open Eyes
  * Closed Eyes
* Accuracy: **95%**

---

## 📈 Drowsiness Detection Method

The system uses **PERCLOS (Percentage of Eye Closure)**.

PERCLOS measures the percentage of time the driver's eyes remain closed within a time window.

If PERCLOS crosses a defined threshold, the system identifies the driver as **drowsy** and triggers an alarm.

---

## 🎯 Future Improvements

* Yawning detection
* Head pose estimation
* Mobile deployment
* Integration with driver monitoring systems
* Edge device deployment (Raspberry Pi)

---


Example:

* Real-time eye detection
* Drowsiness alert system

---

## 📜 License

This project is open-source and available for educational and research purposes.

---

## 👨‍💻 Author

**Ajay Kumar**

Machine Learning Enthusiast
Interested in AI, Computer Vision, and building real-world ML applications.
