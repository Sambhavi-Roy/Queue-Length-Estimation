import sys
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Add SORT path
sys.path.append(r"C:\Users\Sambhavi Roy\Downloads\trackingVehicle\sort")
from sort import Sort  # Import SORT tracker

# --- CONFIGURATION ---
YOLO_LABELS_DIR = r"C:\Users\Sambhavi Roy\Downloads\yolov5-master\yolov5-master\runs\detect\exp6\labels"  # YOLO label folder
VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\720 good.mp4"  # Input video
OUTPUT_VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\stationary_output.mp4"  # Save output video
CSV_OUTPUT_PATH = r"C:\Users\Sambhavi Roy\Downloads\stationary_data.csv"  # Save vehicle data

STATIONARY_THRESHOLD = 5  # Frames to consider a vehicle as stationary
MOVEMENT_THRESHOLD = 2  # Pixel movement threshold

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS

# Initialize SORT tracker
tracker = Sort()

# Store vehicle history
vehicle_positions = {}

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Create DataFrame for stationary vehicle data
columns = ["frame", "vehicle_id", "centroid_x", "centroid_y", "x1", "y1", "x2", "y2"]
df = pd.DataFrame(columns=columns)

# Process each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    label_file = os.path.join(YOLO_LABELS_DIR, f"720 good_{frame_num}.txt")  # Adjust naming pattern

    detections = []
    if os.path.exists(label_file):
        with open(label_file, "r") as file:
            for line in file:
                data = line.strip().split()
                cls, x, y, w, h = map(float, data[0:5])  # YOLO format (normalized)

                # Convert to pixel coordinates
                x1 = int((x - w / 2) * frame_width)
                y1 = int((y - h / 2) * frame_height)
                x2 = int((x + w / 2) * frame_width)
                y2 = int((y + h / 2) * frame_height)

                detections.append([x1, y1, x2, y2, 1])  # Confidence = 1 (dummy)

    # Update tracker
    tracked_objects = tracker.update(np.array(detections))

    # Check for stationary vehicles
    stationary_vehicles = []
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in vehicle_positions:
            vehicle_positions[track_id] = []

        vehicle_positions[track_id].append(centroid)

        # Check movement over last STATIONARY_THRESHOLD frames
        if len(vehicle_positions[track_id]) > STATIONARY_THRESHOLD:
            prev_centroid = vehicle_positions[track_id][-STATIONARY_THRESHOLD]
            movement = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))

            if movement < MOVEMENT_THRESHOLD:
                stationary_vehicles.append(track_id)

    # Display & save results
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        color = (0, 255, 0) if track_id in stationary_vehicles else (255, 0, 0)  # Green if stationary
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save stationary vehicle data
        if track_id in stationary_vehicles:
            df = pd.concat([df, pd.DataFrame([[frame_num, track_id, centroid[0], centroid[1], x1, y1, x2, y2]], columns=columns)], ignore_index=True)

    cv2.imshow("Tracking", frame)
    out.write(frame)  # Save to video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save CSV file with stationary vehicle data
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Stationary vehicle data saved to {CSV_OUTPUT_PATH}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
