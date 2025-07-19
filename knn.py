#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from joblib import load
import os

# === Load model dan encoder ===
MODEL_PATH = os.getenv("KNN_MODEL_PATH", "/home/ros_ws/model_knn_with_report.joblib")

try:
    model_bundle = load(MODEL_PATH)
    model = model_bundle['model']
    encoder = model_bundle['epc_encoder']
    rospy.loginfo("✅ Model dan encoder berhasil dimuat.")
except Exception as e:
    rospy.logerr(f"❌ Gagal memuat model: {e}")
    exit(1)

# === Inisialisasi ROS ===
rospy.init_node("knn_predictor_node", anonymous=False)
pub = rospy.Publisher("/rfid_gol", String, queue_size=10)

# === Callback untuk data dari sensor RFID ===
def callback(msg):
    try:
        epc_raw, rssi_raw = msg.data.strip().split(',')
        epc = epc_raw.strip()
        rssi = float(rssi_raw.strip())

        # Encode EPC ID
        epc_numeric = encoder.transform([epc])[0]
        features = [[epc_numeric, rssi]]

        # Prediksi golongan
        prediction = model.predict(features)[0]
        rospy.loginfo(f"[KNN] EPC: {epc} | RSSI: {rssi} → Golongan: {prediction}")
        pub.publish(prediction)

    except ValueError as ve:
        rospy.logwarn(f"[KNN] Format salah atau EPC tidak dikenal: {msg.data} | {ve}")
    except Exception as e:
        rospy.logerr(f"[KNN] Error saat prediksi: {e}")

# === ROS Subscriber ===
rospy.Subscriber("/rfid_sensor_data", String, callback)
rospy.loginfo("📡 KNN Predictor siap menerima data dari /rfid_sensor_data...")
rospy.spin()