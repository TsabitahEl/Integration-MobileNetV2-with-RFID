# Integration-MobileNetV2-with-RFID

This project demonstrates the integration of AI-based vehicle detection using **MobileNetV2** with **RFID** technology for intelligent decision-making in a smart toll system. The integration is built upon the **Robot Operating System (ROS)** and utilizes **ESP-NOW** protocol and **K-Nearest Neighbors (KNN)** for decision logic.

## 🚀 Key Features

- **High-Accuracy AI Detection (93%)**  
  The pre-trained MobileNetV2 model achieves up to **93% accuracy** in vehicle classification. The model is implemented using the **TensorFlow** framework for reliable and efficient performance.

- **Real-Time Vehicle Detection**  
  MobileNetV2 is optimized for lightweight, real-time vehicle detection via camera input.

- **RFID-Based Identification**  
  Vehicles are equipped with RFID tags. The system reads signal strength (RSSI) to determine proximity.

- **RSSI-Based Decision Logic**  
  - If RSSI values are **-60 dBm, -61 dBm, -62 dBm**, etc., the signal is considered weak and the system switches to **camera-based recognition (MobileNetV2)**.
  - If RSSI is stronger (e.g. > -60 dBm), **RFID-based identification** is used directly.

- **ESP-NOW Communication**  
  Wireless data transmission between ESP32 devices using low-latency ESP-NOW protocol.

- **KNN for Classification**  
  KNN algorithm is implemented for classifying RSSI patterns and assisting decision-making.

## 📦 Technologies Used

- **TensorFlow** (for MobileNetV2)
- **OpenCV** (image processing)
- **Python**
- **ESP32 with ESP-NOW**
- **RFID (RC522 module)**
- **ROS (Robot Operating System)**
- **Scikit-learn** (for KNN)
- **Docker & Docker Compose**

## 📁 Folder Structure (Highlights)

- `run_system.py` – Main script to run the integrated system.
- `bytetrack_mobilenet.py` – Real-time detection and tracking logic.
- `sensor.py` – Reads RFID RSSI data.
- `model_knn.joblib` – Pre-trained KNN model for RSSI classification.
- `docker-compose.yml` – Docker orchestration file.
- `infrence_graph/` – Contains trained MobileNetV2 inference models.

## ⚙️ How It Works

1. The system continuously monitors vehicles approaching the checkpoint.
2. **RFID signal (RSSI)** is captured and analyzed.
3. If the signal is strong (above threshold), it identifies the vehicle via RFID.
4. If the signal is weak, **camera-based detection** (MobileNetV2 + ByteTrack) takes over.
5. All decisions and communications are handled in real-time via **ROS nodes**.

## 📌 Future Improvements

- Integration with cloud-based monitoring dashboards.
- Optimization of MobileNetV2 using TensorRT on Jetson devices.
- Expansion to multi-lane scenarios (MLFF support).

## 🤖 Maintainer

This project is maintained as part of a final project focused on **Multi-Lane Free Flow (MLFF) Smart Tolling Systems**.

---
