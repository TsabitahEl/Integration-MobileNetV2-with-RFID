# Integration-MobileNetV2-with-RFID

This project demonstrates the integration of AI-based vehicle detection using **MobileNetV2** with **RFID** technology for intelligent decision-making in a smart toll system. The system is built upon the **Robot Operating System (ROS)** and utilizes **ESP-NOW** protocol, **K-Nearest Neighbors (KNN)**, and real-time database logging with **Firebase**.

## 🚀 Key Features

- **High-Accuracy AI Detection (93%)**  
  The pre-trained MobileNetV2 model achieves up to **93% accuracy** in vehicle classification. Implemented using the **TensorFlow** framework for efficient and reliable performance.

- **Real-Time Vehicle Detection**  
  MobileNetV2 is optimized for lightweight, real-time detection of vehicles using camera input.

- **RFID-Based Identification**  
  Vehicles are equipped with RFID tags. The system reads signal strength (RSSI) to determine proximity.

- **RSSI-Based Decision Logic**  
  - If RSSI values are **-60 dBm, -61 dBm, -62 dBm**, etc., the signal is considered weak and the system switches to **camera-based recognition (MobileNetV2)**.
  - If RSSI is stronger (e.g. > -60 dBm), **RFID-based identification** is used directly.

- **ESP-NOW Communication**  
  Wireless data transmission between ESP32 devices using low-latency ESP-NOW protocol.

- **KNN for RSSI Classification**  
  KNN algorithm is implemented for classifying RSSI signal patterns and supporting decision-making.

- **🔄 Real-Time Firebase Integration**  
  The system is capable of pushing detection results and identification data directly to **Firebase** in real-time, enabling cloud-based monitoring and analytics.

## 📦 Technologies Used

- **TensorFlow** (for MobileNetV2)
- **OpenCV** (image processing)
- **Python**
- **ESP32 with ESP-NOW**
- **RFID (RC522 module)**
- **ROS (Robot Operating System)**
- **Scikit-learn** (for KNN)
- **Firebase** (Realtime Database)
- **Docker & Docker Compose**

## 📁 Folder Structure (Highlights)

- `run_system.py` – Main script to run the integrated system.
- `bytetrack_mobilenet.py` – Real-time detection and tracking logic.
- `sensor.py` – Reads RFID RSSI data.
- `model_knn.joblib` – Pre-trained KNN model for RSSI classification.
- `firebase_push.py` – Handles real-time data upload to Firebase.
- `docker-compose.yml` – Docker orchestration file.
- `infrence_graph/` – Contains trained MobileNetV2 inference models.

## ⚙️ How It Works

1. The system continuously monitors vehicles approaching the checkpoint.
2. **RFID signal (RSSI)** is captured and analyzed.
3. If the signal is strong (above threshold), the system identifies the vehicle via RFID.
4. If the signal is weak, **camera-based detection** (MobileNetV2 + ByteTrack) takes over.
5. Detection results and data are pushed to **Firebase** in real-time.
6. All decisions and communications are handled in real-time using **ROS nodes**.

## 📌 Future Improvements

- Integration with monitoring dashboards via Firebase Cloud Functions or Grafana.
- Optimization of MobileNetV2 using TensorRT on Jetson devices.
- Expansion to multi-lane scenarios (MLFF support).

## 🤖 Maintainer

This project is maintained as part of a final project focused on **Multi-Lane Free Flow (MLFF) Smart Tolling Systems**, integrating AI, IoT, and cloud-based monitoring.

---
