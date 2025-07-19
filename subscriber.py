import rospy
from std_msgs.msg import String
from datetime import datetime
import requests
import os

# Global state
latest_rfid = None
latest_rssi = None
latest_knn = None
latest_cam = None
last_pushed_epc = None
rssi_threshold = -60  # RSSI cutoff for KNN source

log_file = "/home/ros_ws/logs/subscriber.log"
firebase_base = "https://mlff-e2d86-default-rtdb.firebaseio.com/knn_bitah"

def log(msg):
    print(msg)
    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
        f.flush()

def reset_state():
    global latest_rfid, latest_rssi, latest_knn, latest_cam
    latest_rfid = None
    latest_rssi = None
    latest_knn = None
    latest_cam = None

def push_to_database(source):
    global last_pushed_epc, latest_rfid, latest_rssi, latest_knn, latest_cam
    if latest_rfid == last_pushed_epc:
        log(f"🔁 same tag detected: {latest_rfid}, skipped DB push")
        return

    data = {
        "rssi": latest_rssi,
        "timestamp": datetime.now().isoformat(),
        "source": source
    }

    if source == "knn":
        data["decision"] = latest_knn
        log(f"📥 Both value are fulfilled: pushing to database with KNN source")
        log(f"📡 EPC: {latest_rfid}, RSSI: {latest_rssi}, Decision: {latest_knn}")
    elif source == "camera":
        data["camera_gol"] = latest_cam
        log(f"📥 Both value are fulfilled: pushing to database with Camera source")
        log(f"📡 EPC: {latest_rfid}, RSSI: {latest_rssi}, Camera Gol: {latest_cam}")

    try:
        firebase_url = f"{firebase_base}/{latest_rfid}.json"
        response = requests.put(firebase_url, json=data)
        if response.status_code in (200, 204):
            log("✅ Data pushed to Firebase successfully")
        else:
            log(f"❌ Firebase push failed with status {response.status_code}: {response.text}")
    except Exception as e:
        log(f"❌ Exception during Firebase push: {e}")

    last_pushed_epc = latest_rfid
    reset_state()

def rfid_callback(msg):
    global latest_rfid, latest_rssi
    try:
        parts = msg.data.strip().split(',')
        if len(parts) != 2:
            log(f"❌ Invalid RFID data: {msg.data}")
            return

        epc, rssi = parts[0], float(parts[1])
        log(f"🆕 new tag detected: {epc}, {rssi}")

        if epc != latest_rfid or rssi != latest_rssi:
            latest_rfid = epc
            latest_rssi = rssi
            log(f"✅ update value rfid: {epc}, {rssi}")

            if rssi > rssi_threshold and latest_knn:
                push_to_database("knn")
            elif latest_cam:
                push_to_database("camera")
    except Exception as e:
        log(f"❌ RFID Callback error: {e}")

def knn_callback(msg):
    global latest_knn
    latest_knn = msg.data.strip()
    log(f"📡 received knn decision: {latest_knn}")

    if latest_rfid and latest_rssi and latest_rssi > rssi_threshold:
        push_to_database("knn")

def camera_callback(msg):
    global latest_cam
    msg_clean = msg.data.strip()
    if not msg_clean or msg_clean.lower() == "none" or "|" not in msg_clean:
        log(f"⚠️ Ignored invalid camera_gol message: '{msg_clean}'")
        return

    latest_cam = msg_clean
    log(f"📸 received camera gol: {latest_cam}")
    log(f"✅ updated value camera gol: {latest_cam}")

    if latest_rfid and latest_rssi and latest_rssi <= rssi_threshold:
        push_to_database("camera")

def main():
    rospy.init_node("subscriber_node", anonymous=False)
    rospy.Subscriber("/rfid_sensor_data", String, rfid_callback)
    rospy.Subscriber("/rfid_gol", String, knn_callback)
    rospy.Subscriber("/camera_gol", String, camera_callback)
    log("✅ Subscriber node initialized")
    rospy.spin()

if __name__ == "__main__":
    main()