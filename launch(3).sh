#!/bin/bash

# MLFF System Launcher
echo "===================================================="
echo "🚀 MLFF System Launcher"
echo "===================================================="

# Determine if running on Jetson by checking for CUDA
if [ -d "/usr/local/cuda" ]; then
    DEVICE="cuda:0"
    echo "🖥️ Running on Jetson with CUDA"
else
    DEVICE="cpu"
    echo "🖥️ Running on CPU only"
fi

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Start roscore if not already running
if ! pgrep -x roscore > /dev/null; then
    echo "Starting ROS core..."
    roscore &
    ROSCORE_PID=$!
    sleep 3
    echo "   roscore started (PID: $ROSCORE_PID)"
else
    echo "ROS core is already running"
    ROSCORE_PID=""
fi

# Buat logs directory
mkdir -p logs

# Copy model weight ke direktori YOLOv9
# mkdir -p /home/yolov9
# cp /home/ros_ws/best3_sgd3.pt /home/yolov9/

# Copy model and detection script to MobilenetV2 directory
# mkdir -p /home/mobilenetv2
# cp -r /home/ros_ws/Final_50k/inference_graph/saved_model /home/mobilenetv2/
# cp /home/ros_ws/bismillah_pengujian_fixed.py /home/mobilenetv2/
# cp /home/ros_ws/waskita_ent1.mp4 /home/mobilenetv2/

# Set PYTHONPATH untuk MobilenetV2
export PYTHONPATH=$PYTHONPATH:/home/ros_ws

# Determine source (camera or video)
# if [ -e "/dev/video0" ] && [ "$DEVICE" = "cuda:0" ]; then
#     SOURCE="0"
#     echo "📹 Using camera as source"
# else
#     SOURCE="/home/ros_ws/waskita_baru_1.mp4"
#     echo "🎞️ Using video file"
# fi
# Setup virtual display untuk OpenCV
# echo "Setting up virtual display..."
# Xvfb :99 -screen 0 1024x768x24 &
# export DISPLAY=:99
# sleep 1  # Tunggu sebentar agar Xvfb siap

# Start MobileNetV2 detector dengan parameter yang sesuai
echo "1. Starting MobileNetV2 detector..."
# if [ "$DEVICE" = "cuda:0" ]; then
#     # GPU settings
#     cd /home/mobilenetv2 && python3 bismillah_pengujian.py > /home/ros_ws/logs/detector.log 2>&1 &
# else
    # CPU settings (lighter)
cd /home/ros_ws && python3 bismillah_pengujian_fixed.py > /home/ros_ws/logs/detector.log 2>&1 &
# fi
DETECTOR_PID=$!
echo "1. MobileNet detector started (PID: $DETECTOR_PID)"

# Kembali ke direktori utama
cd /home/ros_ws

# Start sensor reader
echo "2. Starting sensor reader..."
python3 sensor.py > logs/sensor.log 2>&1 &
SENSOR_PID=$!
echo "   Sensor reader started (PID: $SENSOR_PID)"

# Start subscriber with Firebase integration
echo "3. Starting MLFF subscriber with Firebase..."
python3 subscriber.py > logs/subscriber.log 2>&1 &
SUBSCRIBER_PID=$!

echo "4. Starting KNN predictor..."
python3 knn.py > logs/knn.log 2>&1 &
KNN_PID=$!


echo "   MLFF subscriber started (PID: $SUBSCRIBER_PID)"

echo "===================================================="
echo "✅ All MLFF components started successfully!"
echo "Logs are being saved to logs/ directory"
echo "Press Ctrl+C to stop all components"
echo "===================================================="

# Wait a moment for logs to be created
sleep 2

# Show logs in real-time (with error handling)
find logs -type f -name "*.log" | xargs tail -f || echo "No log files found yet"

# Wait for all processes
wait

# Fungsi untuk membersihkan proses saat keluar
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    
    # Hentikan proses berdasarkan PID
    for pid in $SUBSCRIBER_PID $SENSOR_PID $DETECTOR_PID; do
        if ps -p $pid > /dev/null; then
            echo "   Stopping process PID: $pid"
            kill -SIGINT $pid 2>/dev/null || kill -9 $pid 2>/dev/null
        fi
    done
    
    # Hentikan roscore jika kita memulainya
    if [ -n "$ROSCORE_PID" ]; then
        echo "   Stopping roscore"
        kill -SIGINT $ROSCORE_PID 2>/dev/null || kill -9 $ROSCORE_PID 2>/dev/null
    fi
    
    echo "✅ All processes stopped"
    exit 0
}

# Register trap
trap cleanup SIGINT SIGTERM

# Wait for Ctrl+C
echo "Press Ctrl+C to stop all components"
wait