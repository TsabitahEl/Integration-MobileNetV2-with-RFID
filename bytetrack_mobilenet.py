from collections import OrderedDict, deque
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import tensorflow.io as tf_io
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.framework import convert_to_constants
import cv2
import time
import os
import sys
import traceback
from datetime import datetime
import rospy
from std_msgs.msg import String

# Global variables for interactive editing
EDIT_MODE = False
CURRENT_ZONE_POINTS = []
SELECTED_ZONE_TYPE = None  # 'mask' or 'counting'
SELECTED_ZONE_INDEX = None
vehicle_counters = []  # Will be initialized in main
vehicle_pub = None  # ROS publisher

# Define counting and mask zones
# Format: [top-left, top-right, bottom-right, bottom-left]
COUNTING_ZONES = [
    # Lane 1 - Wider counting zone with better coverage
    [[345, 326], [517, 237], [672, 264], [526, 390]], #waskita1_sejajar belakang
    # Lane 2 - Adjusted to cover more area
    [[617, 340], [752, 217], [931, 229], [872, 381]] #waskita1_sejajar belakang
]

# Mask zones expanded to ensure better vehicle detection
MASK_ZONES = [
    # Lane 1 - Larger mask zone
    [[343, 326], [524, 238], [678, 265], [519, 399]], #waskita1_sejajar belakang
    # Lane 2 - Expanded coverage area
    [[617, 341], [753, 216], [930, 229], [876, 386]] #waskita1_sejajar belakang
]

def mouse_callback(event, x, y, flags, param):
    global EDIT_MODE, CURRENT_ZONE_POINTS, SELECTED_ZONE_TYPE, SELECTED_ZONE_INDEX, COUNTING_ZONES, MASK_ZONES
    
    if not EDIT_MODE:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        CURRENT_ZONE_POINTS.append([x, y])
        print(f"Point added: [{x}, {y}]")
        
        # If we complete a polygon (4 points)
        if len(CURRENT_ZONE_POINTS) == 4:
            if SELECTED_ZONE_TYPE == 'mask':
                MASK_ZONES[SELECTED_ZONE_INDEX] = CURRENT_ZONE_POINTS.copy()
            elif SELECTED_ZONE_TYPE == 'counting':
                COUNTING_ZONES[SELECTED_ZONE_INDEX] = CURRENT_ZONE_POINTS.copy()
                
            CURRENT_ZONE_POINTS.clear()
            print(f"Zone {SELECTED_ZONE_TYPE} {SELECTED_ZONE_INDEX} updated!")
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if CURRENT_ZONE_POINTS:
            CURRENT_ZONE_POINTS.pop()
            print("Removed last point")

def draw_editing_overlay(frame):
    if not EDIT_MODE:
        return frame
    
    overlay = frame.copy()
    
    # Draw current editing points
    if CURRENT_ZONE_POINTS:
        # Draw points
        for point in CURRENT_ZONE_POINTS:
            cv2.circle(overlay, (point[0], point[1]), 5, (0, 0, 255), -1)
        
        # Draw lines between points
        for i in range(len(CURRENT_ZONE_POINTS)):
            if i > 0:
                cv2.line(overlay, 
                         tuple(CURRENT_ZONE_POINTS[i-1]), 
                         tuple(CURRENT_ZONE_POINTS[i]), 
                         (0, 0, 255), 2)
    
    # Show edit mode status
    status_text = f"Edit Mode: {SELECTED_ZONE_TYPE} Zone {SELECTED_ZONE_INDEX if SELECTED_ZONE_TYPE else ''}"
    cv2.putText(overlay, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show instructions
    instructions = [
        "Controls:",
        "E - Toggle Edit Mode",
        "M - Edit Mask Zone",
        "C - Edit Counting Zone",
        "R - Reset Current Edit",
        "S - Save Zones",
        "Left Click - Add Point",
        "Right Click - Remove Last Point",
        "Q - Quit"
    ]
    
    y = 60
    for instruction in instructions:
        cv2.putText(overlay, instruction, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    
    return overlay


def save_zone_coordinates():
    """Save current zone coordinates to timestamped files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save counting zones
    counting_zones_file = f"counting_zones_{timestamp}.txt"
    with open(counting_zones_file, 'w') as f:
        f.write("# Counting Zones Coordinates\n")
        f.write("# Format: [top-left, top-right, bottom-right, bottom-left]\n\n")
        f.write("COUNTING_ZONES = [\n")
        for i, zone in enumerate(COUNTING_ZONES):
            f.write(f"    # Zone {i+1}\n")
            f.write(f"    {zone},\n")
        f.write("]\n")
    
    # Save mask zones
    mask_zones_file = f"mask_zones_{timestamp}.txt"
    with open(mask_zones_file, 'w') as f:
        f.write("# Mask Zones Coordinates\n")
        f.write("# Format: [top-left, top-right, bottom-right, bottom-left]\n\n")
        f.write("MASK_ZONES = [\n")
        for i, zone in enumerate(MASK_ZONES):
            f.write(f"    # Zone {i+1}\n")
            f.write(f"    {zone},\n")
        f.write("]\n")
    
    print(f"Zone coordinates saved:")
    print(f"  Counting zones: {counting_zones_file}")
    print(f"  Mask zones: {mask_zones_file}")
    
    return counting_zones_file, mask_zones_file

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Video settings
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
VIDEO_FPS = 60

# Output settings
OUTPUT_DIR = "output_videos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class ZoneNormalizer:
    def __init__(self, reference_image_path):
        """Initialize with the reference image (the one used to create coordinates)"""
        ref_img = cv2.imread(reference_image_path)
        self.ref_height, self.ref_width = ref_img.shape[:2]
        
    def normalize_coordinates(self, coordinates):
        """Convert absolute coordinates to relative (0-1) coordinates"""
        return [
            [x / self.ref_width, y / self.ref_height]
            for x, y in coordinates
        ]
    
    def denormalize_coordinates(self, normalized_coords, target_width, target_height):
        """Convert relative coordinates back to absolute for the target image size"""
        return [
            [int(x * target_width), int(y * target_height)]
            for x, y in normalized_coords
        ]
    
def load_model(model_path):
    try:
        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            return None
            
        # Load saved model
        print("Loading saved model...")
        detect_fn = tf.saved_model.load(model_path)
        print("Model loaded successfully")
        return detect_fn
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def display_model_summary(detect_fn):
    print("\nGenerating model summary...")
    # Extract the concrete function
    concrete_func = detect_fn.signatures["serving_default"]
    
    # Convert the model to a frozen graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    
    # Count the number of layers (operations in the graph)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print(f"Number of layers: {len(layers)}")
    
    # Count parameters from the frozen function graph
    total_parameters = 0
    for operation in frozen_func.graph.get_operations():
        if operation.type.lower() in ['const']:
            try:
                tensor = operation.outputs[0]
                shape = tensor.shape
                if shape.rank is not None:  # Check if shape is known
                    params = 1
                    for dim in shape:
                        if dim is not None:  # Check if dimension is known
                            params *= dim
                    total_parameters += params
            except:
                continue
    
    print(f"Number of parameters: {total_parameters:,}")
    print("Model summary complete.")

# UI Constants
margin = 10
header_height = 40
line_height = 30

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def detect_objects(bgr_frame, detect_fn, category_index, vehicle_counters, counting_zones, mask_zones):
    try:
        # Reset frame tracking
        for counter in vehicle_counters:
            counter.reset_frame_tracking()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        start_time = time.time()
        detections = detect_fn(input_tensor)
        inference_time = time.time() - start_time

        # Extract detection data
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Get detection arrays
        boxes = detections['detection_boxes']
        classes = detections['detection_classes'].astype(np.int64)
        scores = detections['detection_scores']

        # Filter detections
        min_score_thresh = 0.5
        selected_indices = np.where(scores > min_score_thresh)[0]

        height, width = bgr_frame.shape[:2]

        # Color map for classes
        color_map = {1: (0, 0, 255), 2: (203, 192, 255), 3: (144, 238, 144), 
                    4: (0, 165, 255), 5: (255, 255, 0)}

        # Create diagonal counting lines from counting zones
        counting_lines = []
        for zone in counting_zones:
            # Create diagonal line from top-left to bottom-right
            tl = zone[0]  # top-left
            br = zone[2]  # bottom-right
            
            # Diagonal line
            counting_lines.append([tl[0], tl[1], br[0], br[1]])

        # Prepare detections for ByteTrack
        zone_detections = [[] for _ in range(len(counting_zones))]

        # Process and draw detections
        for i in selected_indices:
            # Get normalized box coordinates
            box = boxes[i]
            
            # Convert to pixel coordinates
            y1, x1 = int(box[0] * height), int(box[1] * width)
            y2, x2 = int(box[2] * height), int(box[3] * width)
            
            # Get class ID and confidence
            class_id = classes[i]
            confidence = scores[i]
            color = color_map.get(class_id, (0, 0, 255))
            
            # Draw bounding box
            cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add class label and confidence
            if class_id in category_index:
                class_name = category_index[class_id]['name']
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(bgr_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check which zone the detection belongs to
                for zone_idx, zone in enumerate(counting_zones):
                    if point_in_polygon((center_x, center_y), zone):
                        # Format detection for ByteTrack
                        detection = [x1, y1, x2, y2, confidence, class_id]
                        zone_detections[zone_idx].append(detection)
                        break

        # Update ByteTrack counters and count vehicles
        for zone_idx, detections_in_zone in enumerate(zone_detections):
            if detections_in_zone:
                # Convert to numpy array
                detections_array = np.array(detections_in_zone)
                
                # Update tracks
                vehicle_counters[zone_idx].update_tracks(detections_array)
                
                # Count line crossings
                if len(counting_lines) > zone_idx:
                    new_counts = vehicle_counters[zone_idx].count_line_crossings(counting_lines[zone_idx])
                    if new_counts > 0:
                        print(f"Lane {zone_idx + 1}: {new_counts} new vehicles counted")

        # Draw zones and counting lines
        for i, zone in enumerate(counting_zones):
            zone_array = np.array(zone, np.int32)
            cv2.polylines(bgr_frame, [zone_array], True, (0, 0, 255), 2)
            cv2.putText(bgr_frame, f"Lane {i+1}", (zone[0][0], zone[0][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw counting line
            if i < len(counting_lines):
                line = counting_lines[i]
                cv2.line(bgr_frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
                # Draw arrow to show counting direction
                # arrow_mid_x = (line[0] + line[2]) // 2
                # arrow_mid_y = (line[1] + line[3]) // 2
                # cv2.arrowedLine(bgr_frame, 
                #               (arrow_mid_x, arrow_mid_y), 
                #               (arrow_mid_x + 20, arrow_mid_y + 20),
                #               (0, 255, 0), 2, tipLength=0.3)

        # Draw mask zones
        for i, zone in enumerate(mask_zones):
            zone_array = np.array(zone, np.int32)
            overlay = bgr_frame.copy()
            cv2.fillPoly(overlay, [zone_array], (0, 255, 0, 64))  # Semi-transparent green
            cv2.addWeighted(overlay, 0.2, bgr_frame, 0.8, 0, bgr_frame)

        return bgr_frame, inference_time

    except Exception as e:
        print(f"Error in detect_objects: {e}")
        traceback.print_exc()
        return bgr_frame, 0

def setup_ros():
    """Initialize ROS node and publisher"""
    try:
        rospy.init_node('vehicle_detection_bytetrack', anonymous=True)
        global vehicle_pub
        vehicle_pub = rospy.Publisher('/camera_gol', String, queue_size=10)
        print("🚀 ROS publisher initialized")
        print("Publisher topic: /camera_gol")
        return True
    except Exception as e:
        print(f"⚠️ ROS initialization error: {e}")
        return False

def publish_to_ros(golongan, jalur):
    """Publish detection to ROS topic"""
    global vehicle_pub
    if vehicle_pub is not None:
        try:
            msg_data = f"G{golongan}|{jalur}"
            vehicle_pub.publish(String(msg_data))
            print(f"📤 Published to ROS: {msg_data}")
        except Exception as e:
            print(f"⚠️ ROS publish error: {e}")

def main():
    # Declare global variables that will be modified
    global EDIT_MODE, CURRENT_ZONE_POINTS, SELECTED_ZONE_TYPE, SELECTED_ZONE_INDEX, vehicle_counters
    
    try:
        print("\nInitializing object detection with GPU acceleration...")
        print("TensorFlow version:", tf.__version__)
        print("Python version:", sys.version)
        
        print("\nLoading model... This might take a few minutes.")
        model_path = "D:\Deploy_Bitah\Final_50k\inference_graph\saved_model"
        label_map_path = "D:\Deploy_Bitah\Final_50k\objects_label_map.pbtxt"

        # Load model
        detect_fn = load_model(model_path)
        if detect_fn is None:
            print("Failed to load model")
            return

        # Load label map
        category_index = label_map_util.create_category_index_from_labelmap(
            label_map_path, use_display_name=True)
        
        # Display model summary
        display_model_summary(detect_fn)

        # Initialize ROS publisher
        setup_ros()
        # Process video
        video_path = "D:\Deploy_Bitah\Sample\waskita_ent1.mp4"
        print(f"Processing video from: {os.path.abspath(video_path)}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(0)
        # Set resolusi yang diinginkan
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Lebar 1920 pixels
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Tinggi 1080 pixels
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        cv2.namedWindow('Object Detection')
        cv2.setMouseCallback('Object Detection', mouse_callback)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
        
        # Prepare output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"countingBitahWas_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_time = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
                
            frame_count += 1
            frame_start = time.time()
            
            # Process frame (input is BGR frame from OpenCV)
            processed_frame, inference_time = detect_objects(frame, detect_fn, category_index, vehicle_counters, COUNTING_ZONES, MASK_ZONES)

            # Calculate FPS
            frame_time = time.time() - frame_start
            total_time += frame_time
            avg_fps = frame_count / total_time
            
            # Add statistics overlay to frame (BEFORE keyboard handling)
            margin = 5
            header_height = 25
            line_height = 15
            overlay_width = 300  # Sesuaikan dengan ukuran overlay Anda
            
            # Calculate total_height for overlay background
            total_height = header_height + (line_height * 7) + 20  # Tambah height untuk G1-G5

            # Create semi-transparent background for entire summary
            overlay = np.zeros((total_height, 300, 3), dtype=np.uint8)
            cv2.rectangle(overlay, (0, 0), (300, total_height), (40, 40, 40), -1)

            # Ensure we don't go out of frame bounds
            if margin + total_height < processed_frame.shape[0] and margin + 300 < processed_frame.shape[1]:
                region = processed_frame[margin:margin + total_height, margin:margin + 300]
                if region.shape[:2] == overlay.shape[:2]:
                    cv2.addWeighted(overlay, 0.7, region, 0.3, 0, region)
                    processed_frame[margin:margin + total_height, margin:margin + 300] = region

            # Add vehicle count summary header
            header_text = "Vehicle Count Summary"
            cv2.putText(
                processed_frame,
                header_text,
                (margin, margin + header_height - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw horizontal separator line
            cv2.line(
                processed_frame,
                (margin + 5, margin + header_height),
                (margin + overlay_width - 5, margin + header_height),
                (255, 255, 255),
                2
            )

            # Draw vertical separator line
            # cv2.line(
            #     processed_frame,
            #     (margin + 205, margin + header_height + 5),
            #     (margin + 205, margin + total_height - 5),
            #     (255, 255, 255),
            #     1
            # )
            

            # Definisikan lebar untuk setiap lane
            half_width = overlay_width // 2  # Membagi overlay menjadi 2 kolom sama lebar

            # Koordinat untuk kotak lane 1
            lane1_x_start = margin
            lane1_x_end = margin + half_width

            # Koordinat untuk kotak lane 2
            lane2_x_start = margin + half_width
            lane2_x_end = margin + overlay_width

            # Gambar garis vertikal TEPAT di antara kedua kotak lane
            cv2.line(
                processed_frame,
                (lane2_x_start, margin + header_height),  # Titik atas - SAMA dengan awal lane2
                (lane2_x_start, margin + total_height),   # Titik bawah
                (255, 255, 255),
                2
            )
            # Draw counts for each lane side by side
            lane_colors = [(0, 165, 255), (255, 255, 0)]  # Orange dan Cyan dalam BGR
            lane_bg_colors = [(30, 90, 150), (100, 100, 0)]

            for i in range(2):
                if i == 0:
                    # Lane 1
                    x_start = lane1_x_start
                    x_end = lane1_x_end
                else:
                    # Lane 2
                    x_start = lane2_x_start
                    x_end = lane2_x_end
                
                y_start = margin + header_height
                y_end = margin + total_height
                
                # Background untuk lane
                cv2.rectangle(processed_frame,
                            (x_start, y_start),
                            (x_end, y_end),
                            lane_bg_colors[i], -1)
                
                # Text untuk lane (pastikan padding konsisten)
                text_padding_x = 10  # Padding horizontal untuk text
                text_padding_y = 20  # Padding vertical untuk text pertama
                
                cv2.putText(
                    processed_frame,
                    f"Lane {i + 1}: {vehicle_counters[i].total_count}",
                    (x_start + text_padding_x, y_start + text_padding_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    lane_colors[i],
                    1
                )
                
                # Golongan dengan padding konsisten
                item_spacing = 18  # Jarak antar golongan
                
                for g in range(1, 6):
                    class_name = f"G{g}"
                    count = vehicle_counters[i].class_counts.get(class_name, 0)
                    
                    cv2.putText(
                        processed_frame,
                        f"{class_name}: {count}",
                        (x_start + text_padding_x, y_start + text_padding_y + (g * item_spacing)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        lane_colors[i],
                        1
                    )

            # Add FPS and inference time below the overlay box
            # fps_y_pos = margin + total_height + 25
            # cv2.putText(
            #     processed_frame,
            #     f"FPS: {avg_fps:.1f}",
            #     (margin + 10, fps_y_pos),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0),  
            #     2
            # )
            
            # cv2.putText(
            #     processed_frame,
            #     f"Inference: {inference_time*1000:.1f}ms",
            #     (margin + 10, fps_y_pos + line_height),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0,)  
            #     2
            # )
            # Posisikan FPS dan inference di sebelah kanan vehicle summary
            stats_width = 220  # Lebar area FPS dan inference
            stats_height = 60  # Tinggi area
            line_height = 25   # Spasi antar baris

            # Hitung posisi X agar di pojok kanan dengan margin
            fps_x_pos = processed_frame.shape[1] - stats_width - 10  # 10 pixel dari tepi kanan
            fps_y_pos = 30  # Posisi Y dari atas

            # Buat mini background untuk FPS dan inference
            stats_overlay = np.zeros((stats_height, stats_width, 3), dtype=np.uint8)
            cv2.rectangle(stats_overlay, (0, 0), (stats_width, stats_height), (40, 40, 40), -1)

            # Pastikan tidak keluar dari frame
            if fps_x_pos >= 0 and fps_y_pos + stats_height <= processed_frame.shape[0]:
                stats_region = processed_frame[10:10 + stats_height, fps_x_pos:fps_x_pos + stats_width]
                if stats_region.shape[:2] == stats_overlay.shape[:2]:
                    cv2.addWeighted(stats_overlay, 0.7, stats_region, 0.3, 0, stats_region)
                    processed_frame[10:10 + stats_height, fps_x_pos:fps_x_pos + stats_width] = stats_region

            # Tulis FPS dan inference time
            cv2.putText(
                processed_frame,
                f"FPS: {avg_fps:.1f}",
                (fps_x_pos + 10, fps_y_pos + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green color
                2
            )

            cv2.putText(
                processed_frame,
                f"Inference: {inference_time*1000:.1f}ms",
                (fps_x_pos + 10, fps_y_pos + line_height + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green color
                2
            ) 
            # Handle edit mode overlay
            if EDIT_MODE:
                processed_frame = draw_editing_overlay(processed_frame)
            
            # Write and display frame
            out.write(processed_frame)
            cv2.imshow('Object Detection', processed_frame)
            
            # Handle keyboard input once per frame
            key = cv2.waitKey(1) & 0xFF
            
            # Handle edit mode first
            if key == ord('e'):  # Toggle edit mode
                EDIT_MODE = not EDIT_MODE
                if EDIT_MODE:
                    SELECTED_ZONE_TYPE = None
                    SELECTED_ZONE_INDEX = None
                    print("Edit mode ON - Press 'm' for mask zones or 'c' for counting zones")
                else:
                    CURRENT_ZONE_POINTS.clear()
                    SELECTED_ZONE_TYPE = None
                    SELECTED_ZONE_INDEX = None
                    print("Edit mode OFF")
            elif EDIT_MODE and key == ord('m'):  # Edit mask zone
                SELECTED_ZONE_TYPE = 'mask'
                if SELECTED_ZONE_INDEX is None or SELECTED_ZONE_TYPE != 'mask':
                    SELECTED_ZONE_INDEX = 0
                else:
                    SELECTED_ZONE_INDEX = (SELECTED_ZONE_INDEX + 1) % len(MASK_ZONES)
                CURRENT_ZONE_POINTS.clear()
                print(f"Editing mask zone {SELECTED_ZONE_INDEX + 1} of {len(MASK_ZONES)}")
            elif EDIT_MODE and key == ord('c'):  # Edit counting zone
                SELECTED_ZONE_TYPE = 'counting'
                if SELECTED_ZONE_INDEX is None or SELECTED_ZONE_TYPE != 'counting':
                    SELECTED_ZONE_INDEX = 0
                else:
                    SELECTED_ZONE_INDEX = (SELECTED_ZONE_INDEX + 1) % len(COUNTING_ZONES)
                CURRENT_ZONE_POINTS.clear()
                print(f"Editing counting zone {SELECTED_ZONE_INDEX + 1} of {len(COUNTING_ZONES)}")
            elif EDIT_MODE and key == ord('r'):  # Reset current zone editing
                CURRENT_ZONE_POINTS.clear()
                print("Zone editing reset - cleared all points")
            elif EDIT_MODE and key == ord('s'):  # Save zones
                counting_file, mask_file = save_zone_coordinates()
                print("Zones saved successfully!")
            # Check for quit
            elif key == ord('q') or key == 27:  # 'q' key or ESC
                print("\nProcessing finished! Generating final statistics...")
                
                # Save zone coordinates before quitting
                if EDIT_MODE or any(COUNTING_ZONES) or any(MASK_ZONES):
                    print("\nSaving zone coordinates...")
                    counting_file, mask_file = save_zone_coordinates()
                
                # Calculate final averages
                avg_inference = total_time/frame_count * 1000
                avg_fps = frame_count/total_time
                
                # Print final statistics with more detail
                print("\n=== Processing Statistics ===")
                print(f"Total frames processed: {frame_count}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Average inference time: {avg_inference:.2f}ms")
                print(f"Total processing time: {total_time:.2f} seconds")
                print("\n=== Vehicle Count Summary ===")
                for i, counter in enumerate(vehicle_counters):
                    print(f"\nLane {i+1} Summary:")
                    print(f"Total vehicles: {counter.total_count}")
                    if counter.total_count > 0:
                        for cls, count in counter.class_counts.items():
                            percentage = (count / counter.total_count) * 100
                            print(f"{cls}: {count} ({percentage:.1f}%)")
                
                # Print zone coordinates recap
                print("\n=== Zone Coordinates Recap ===")
                print("Counting Zones:")
                for i, zone in enumerate(COUNTING_ZONES):
                    print(f"  Zone {i+1}: {zone}")
                print("Mask Zones:")
                for i, zone in enumerate(MASK_ZONES):
                    print(f"  Zone {i+1}: {zone}")
                    
                print(f"\nOutput video saved to: {output_path}")
                
                # Save detailed statistics to file
                stats_path = "processing_stats.txt"
                with open(stats_path, "w") as f:
                    f.write("=== Processing Statistics ===\n\n")
                    f.write(f"Video Properties:\n")
                    f.write(f"- Width: {width}\n")
                    f.write(f"- Height: {height}\n")
                    f.write(f"- FPS: {fps}\n\n")
                    f.write(f"Performance Metrics:\n")
                    f.write(f"- Total frames: {frame_count}\n")
                    f.write(f"- Average FPS: {avg_fps:.2f}\n")
                    f.write(f"- Average inference time: {avg_inference:.2f}ms\n")
                    f.write(f"- Total time: {total_time:.2f}s\n\n")
                    f.write(f"Vehicle Counts:\n")
                    for i, counter in enumerate(vehicle_counters):
                        f.write(f"Lane {i+1}:\n")
                        f.write(f"- Total vehicles: {counter.total_count}\n")                        
                        if counter.total_count > 0:
                            for cls, count in counter.class_counts.items():
                                percentage = (count / counter.total_count) * 100
                                f.write(f"- {cls}: {count} ({percentage:.1f}%)\n")
                        else:
                            f.write("- No vehicles detected\n")
                        f.write("\n")
                    f.write(f"Zone Coordinates:\n")
                    f.write(f"Counting Zones:\n")
                    for i, zone in enumerate(COUNTING_ZONES):
                        f.write(f"- Zone {i+1}: {zone}\n")
                    f.write(f"Mask Zones:\n")
                    for i, zone in enumerate(MASK_ZONES):
                        f.write(f"- Zone {i+1}: {zone}\n")
                    f.write(f"\nOutput saved to: {output_path}\n")
                print(f"Detailed statistics saved to: {stats_path}")
                break
            
            # Add statistics overlay to frame
            margin = 5
            header_height = 25
            line_height = 15
            
            # Calculate total_height for overlay background
            total_height = header_height + (line_height * 7) + 30  # Tambah height untuk G1-G5

            # Create semi-transparent background for entire summary
            overlay = np.zeros((total_height, 410, 3), dtype=np.uint8)
            cv2.rectangle(overlay, (0, 0), (410, total_height), (40, 40, 40), -1)

            # Ensure we don't go out of frame bounds
            if margin + total_height < processed_frame.shape[0] and margin + 410 < processed_frame.shape[1]:
                region = processed_frame[margin:margin + total_height, margin:margin + 410]
                if region.shape[:2] == overlay.shape[:2]:
                    cv2.addWeighted(overlay, 0.7, region, 0.3, 0, region)
                    processed_frame[margin:margin + total_height, margin:margin + 410] = region

            # Add vehicle count summary header
            header_text = "Vehicle Count Summary"
            cv2.putText(
                processed_frame,
                header_text,
                (margin + 10, margin + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Draw horizontal separator line
            cv2.line(
                processed_frame,
                (margin + 5, margin + header_height),
                (margin + 405, margin + header_height),
                (255, 255, 255),
                2
            )

            # Draw vertical separator line
            cv2.line(
                processed_frame,
                (margin + 205, margin + header_height + 5),
                (margin + 205, margin + total_height - 5),
                (255, 255, 255),
                1
            )

            # Draw counts for each lane side by side
            lane_colors = [(0, 165, 255), (255, 255, 0)]  # Orange dan Cyan dalam BGR
            lane_bg_colors = [(30, 90, 150), (100, 100, 0)]

            for i in range(2):
                # Create background rectangle for each lane
                x_start = margin + (i * 200) + 10
                y_start = margin + header_height + 5
                x_end = x_start + 185
                y_end = margin + total_height - 5
                
                # Draw colored background for each lane
                cv2.rectangle(processed_frame,
                            (x_start, y_start),
                            (x_end, y_end),
                            lane_bg_colors[i], -1)
                
                # Draw lane total
                cv2.putText(
                    processed_frame,
                    f"Lane {i + 1}: {vehicle_counters[i].total_count}",
                    (x_start + 10, y_start + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    lane_colors[i],
                    2
                )
                
                # Draw counts for each golongan
                y_offset = y_start + 60  # Mulai di bawah total count
                for g in range(1, 6):  # G1 sampai G5
                    class_name = f"G{g}"
                    count = vehicle_counters[i].class_counts.get(class_name, 0)
                    
                    # Only show count if > 0 or show 0 for consistency
                    cv2.putText(
                        processed_frame,
                        f"{class_name}: {count}",
                        (x_start + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        lane_colors[i],
                        2
                    )
                    y_offset += 25  # Spasi antar golongan

            # Add FPS and inference time below the overlay box
            fps_y_pos = margin + total_height + 25
            cv2.putText(
                processed_frame,
                f"FPS: {avg_fps:.1f}",
                (margin + 10, fps_y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            cv2.putText(
                processed_frame,
                f"Inference: {inference_time*1000:.1f}ms",
                (margin + 10, fps_y_pos + line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

class VehicleCounter:
    def __init__(self, lane_id=0):
        self.lane_id = lane_id
        self.total_count = 0
        self.class_counts = {'G1': 0, 'G2': 0, 'G3': 0, 'G4': 0, 'G5': 0}
        self.tracked_vehicles = set()  # Track vehicles to prevent duplicate counting
        self.frame_vehicles = {}  # Track vehicles per frame for duplicate prevention

    def increment(self, class_name, vehicle_id=None):
        if class_name in self.class_counts:
            # If vehicle_id is provided, check for duplicates
            if vehicle_id is not None:
                if vehicle_id in self.tracked_vehicles:
                    return False  # Already counted this vehicle
                self.tracked_vehicles.add(vehicle_id)
            
            self.total_count += 1
            self.class_counts[class_name] += 1
            
            # Debug output
            print(f"🚗 Lane {self.lane_id + 1}: {class_name} counted! Total: {self.total_count}")
            return True
        return False

    def reset_frame_tracking(self):
        """Reset frame-based tracking (call once per frame)"""
        self.frame_vehicles.clear()

    def is_duplicate_in_frame(self, detection_key):
        """Check if this detection was already processed in current frame"""
        if detection_key in self.frame_vehicles:
            return True
        self.frame_vehicles[detection_key] = True
        return False

    def get_percentages(self):
        if self.total_count == 0:
            return {}
        return {cls: (count / self.total_count) * 100 
                for cls, count in self.class_counts.items()}

    def get_active_classes(self):
        """Return only classes that have count > 0"""
        return {cls: count for cls, count in self.class_counts.items() if count > 0}

class ByteTrackVehicleCounter:
    """Advanced ByteTrack vehicle counter with line crossing detection"""
    
    def __init__(self, lane_id, log_file=None):
        self.lane_id = lane_id
        self.log_file = log_file or f"bytetrack_lane_{lane_id}_counts.csv"
        
        # Track management with default parameters
        self.track_thresh = 0.4
        self.track_buffer = 20
        self.match_thresh = 0.6
        self.min_box_area = 15
        
        # Track storage
        self.tracked_tracks = OrderedDict()
        self.lost_tracks = OrderedDict()
        self.frame_id = 0
        self.track_id_count = 0
        
        # Vehicle counting (adapted for G1-G5 system)
        self.total_count = 0
        self.class_counts = {"G1": 0, "G2": 0, "G3": 0, "G4": 0, "G5": 0}
        self.counted_tracks = set()
        self.class_names = {1: "G1", 2: "G2", 3: "G3", 4: "G4", 5: "G5"}
        
        # Performance tracking
        self.track_history_length = 8
        
        # Initialize CSV log
        self._initialize_log()
        
        print(f"🚗 ByteTrack Counter initialized for Lane {lane_id+1}")
    
    def _initialize_log(self):
        """Initialize CSV log file"""
        try:
            with open(self.log_file, 'w') as f:
                f.write("Timestamp,Lane,Total,G1,G2,G3,G4,G5,G1%,G2%,G3%,G4%,G5%\n")
        except Exception as e:
            print(f"⚠️  Could not initialize log file: {e}")
    
    def calculate_iou(self, box1, box2):
        """Fast IoU calculation optimized for tracking"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detections, roi_mask=None):
        """Main ByteTrack update"""
        self.frame_id += 1
        
        if len(detections) == 0:
            self._age_tracks()
            return []
        
        # Convert detections to proper format if needed
        if isinstance(detections, list):
            # Convert from your detection format
            processed_dets = []
            for det in detections:
                # det format: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'score': conf, 'class': cls, 'class_name': class_name}
                bbox = [det['x1'], det['y1'], det['x2'], det['y2'], det['score'], det['class']]
                processed_dets.append(bbox)
            detections = np.array(processed_dets)
        
        if len(detections) == 0:
            self._age_tracks()
            return []
        
        # Get current track data
        track_boxes = []
        track_ids = []
        for track_id, track in self.tracked_tracks.items():
            track_boxes.append(track['bbox'])
            track_ids.append(track_id)
        
        # Association using IoU matching
        if track_boxes and len(detections) > 0:
            det_boxes = detections[:, :4]
            
            # Calculate cost matrix
            cost_matrix = np.zeros((len(track_boxes), len(det_boxes)))
            for i, track_box in enumerate(track_boxes):
                for j, det_box in enumerate(det_boxes):
                    iou = self.calculate_iou(track_box, det_box)
                    cost_matrix[i, j] = 1 - iou  # Convert to cost
            
            # Greedy assignment
            matches = []
            matched_tracks = set()
            matched_dets = set()
            
            for i in range(len(track_boxes)):
                for j in range(len(det_boxes)):
                    if (i not in matched_tracks and j not in matched_dets and 
                        cost_matrix[i, j] < (1 - self.match_thresh)):
                        matches.append((i, j))
                        matched_tracks.add(i)
                        matched_dets.add(j)
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                track_id = track_ids[track_idx]
                det = detections[det_idx]
                self._update_track(track_id, det)
            
            # Create new tracks for unmatched detections
            for j in range(len(detections)):
                if j not in matched_dets:
                    self._create_track(detections[j])
            
            # Move unmatched tracks to lost
            for i in range(len(track_ids)):
                if i not in matched_tracks:
                    track_id = track_ids[i]
                    self.lost_tracks[track_id] = self.tracked_tracks[track_id]
                    del self.tracked_tracks[track_id]
        else:
            # No existing tracks, create new ones
            for det in detections:
                self._create_track(det)
        
        # Clean up old tracks
        self._age_tracks()
        
        return list(self.tracked_tracks.values())
    
    def _create_track(self, detection):
        """Create new track from detection"""
        self.track_id_count += 1
        bbox = detection[:4]
        conf = float(detection[4])
        cls = int(detection[5])
        
        # Filter by confidence and size
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if conf >= self.track_thresh and box_area >= self.min_box_area:
            track = {
                'track_id': self.track_id_count,
                'bbox': bbox,
                'score': conf,
                'class': cls,
                'age': 1,
                'time_since_update': 0,
                'history': deque([bbox], maxlen=self.track_history_length),
                'confirmed': False
            }
            self.tracked_tracks[self.track_id_count] = track
    
    def _update_track(self, track_id, detection):
        """Update existing track"""
        track = self.tracked_tracks[track_id]
        bbox = detection[:4]
        conf = float(detection[4])
        cls = int(detection[5])
        
        track['bbox'] = bbox
        track['score'] = conf
        track['class'] = cls
        track['age'] += 1
        track['time_since_update'] = 0
        track['history'].append(bbox)
        
        # Confirm track after minimum observations
        if track['age'] >= 3:
            track['confirmed'] = True
    
    def _age_tracks(self):
        """Age tracks and remove old ones"""
        # Age tracked tracks
        for track_id in list(self.tracked_tracks.keys()):
            track = self.tracked_tracks[track_id]
            track['time_since_update'] += 1
            
            # Move to lost if not updated
            if track['time_since_update'] > 1:
                self.lost_tracks[track_id] = track
                del self.tracked_tracks[track_id]
        
        # Remove old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            track = self.lost_tracks[track_id]
            track['time_since_update'] += 1
            
            if track['time_since_update'] > self.track_buffer:
                del self.lost_tracks[track_id]
    
    def count_line_crossings(self, counting_line):
        """Count vehicles crossing the counting line"""
        if not counting_line or len(counting_line) != 4:
            return 0
        
        x1, y1, x2, y2 = counting_line
        new_counts = 0
        
        for track_id, track in self.tracked_tracks.items():
            # Only count confirmed tracks that haven't been counted
            if (not track['confirmed'] or 
                track_id in self.counted_tracks or 
                len(track['history']) < 2):
                continue
            
            # Get last two positions
            history = list(track['history'])
            prev_bbox = history[-2]
            curr_bbox = history[-1]
            
            # Calculate center points
            prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
            curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
            
            # Check line intersection
            if self._line_intersection(prev_center, curr_center, (x1, y1), (x2, y2)):
                vehicle_class = track['class']
                class_name = self.class_names.get(vehicle_class, f"G{vehicle_class}")
                
                if class_name in self.class_counts:
                    self.class_counts[class_name] += 1
                    self.total_count += 1
                    self.counted_tracks.add(track_id)
                    new_counts += 1
                    
                    # Log the count
                    self._log_count()
                    
                    print(f"🚗 Lane {self.lane_id+1}: {class_name} counted (Track {track_id})")
                    # Extract the golongan (e.g. 1 from "G1")
                    golongan = class_name[1] if class_name.startswith("G") and len(class_name) > 1 else class_name
                    jalur = self.lane_id + 1
                    publish_to_ros(golongan, jalur)
        
        return new_counts
    
    def _line_intersection(self, p1, p2, p3, p4):
        """Check if line segment p1p2 intersects with line segment p3p4"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def get_percentages(self):
        """Calculate class percentages"""
        if self.total_count == 0:
            return {"G1": 0.0, "G2": 0.0, "G3": 0.0, "G4": 0.0, "G5": 0.0}
        return {cls: (self.class_counts[cls] / self.total_count) * 100 for cls in self.class_counts}
    
    def _log_count(self):
        """Log count to CSV file"""
        try:
            percentages = self.get_percentages()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp},{self.lane_id+1},{self.total_count}"
            
            # Add counts
            for class_name in ["G1", "G2", "G3", "G4", "G5"]:
                log_line += f",{self.class_counts[class_name]}"
            
            # Add percentages
            for class_name in ["G1", "G2", "G3", "G4", "G5"]:
                log_line += f",{percentages[class_name]:.1f}"
            
            with open(self.log_file, 'a') as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"⚠️  Logging error: {e}")

    # Keep compatibility with existing code
    def reset_frame_tracking(self):
        """Reset frame-based tracking (compatibility method)"""
        pass

    def increment(self, class_name, vehicle_id=None):
        """Legacy method for compatibility"""
        if class_name in self.class_counts:
            self.total_count += 1
            self.class_counts[class_name] += 1
            print(f"🚗 Lane {self.lane_id + 1}: {class_name} counted! Total: {self.total_count}")
            return True
        return False

if __name__ == "__main__":
    # Initialize ByteTrack vehicle counters
    vehicle_counters = [
        ByteTrackVehicleCounter(lane_id=0, log_file="lane1_counts.csv"),  # Lane 1 
        ByteTrackVehicleCounter(lane_id=1, log_file="lane2_counts.csv")   # Lane 2
    ]
    
    # Setup ROS
    ros_initialized = setup_ros()
    if not ros_initialized:
        print("Warning: ROS not initialized. Some features may not work.")

    main()
