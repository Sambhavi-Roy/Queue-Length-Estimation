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
YOLO_LABELS_DIR = r"C:\Users\Sambhavi Roy\Downloads\yolov5-master\yolov5-master\runs\detect\exp11\labels"
VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\sample 9\video 9.mp4"
OUTPUT_VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\sample 9\9speed_output.mp4"
CSV_OUTPUT_PATH = r"C:\Users\Sambhavi Roy\Downloads\sample 9\9speed_data.csv"

GSD = 0.1048  # Ground Sampling Distance (meters per pixel) at 100m altitude
FPS = 30  # Frames per second of video

# Reference resolution where polygon was defined1079
REF_WIDTH, REF_HEIGHT = 1540, 1079

road_polygon_ref = np.array([
    [200, 206],
    [274, 168],
    [1512, 972],
    [1254, 1060],
], np.int32)


# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS

# Scale road polygon to match actual video frame size
scale_x = frame_width / REF_WIDTH
scale_y = frame_height / REF_HEIGHT
road_polygon = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in road_polygon_ref], np.int32)

# Initialize SORT tracker
tracker = Sort()

# Store vehicle history
vehicle_positions = {}
vehicle_speeds = {}

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Create DataFrame for speed data
columns = ["frame", "vehicle_id", "speed_kmh", "centroid_x", "centroid_y", "x1", "y1", "x2", "y2"]
df = pd.DataFrame(columns=columns)

# Process each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    label_file = os.path.join(YOLO_LABELS_DIR, f"video 9_{frame_num}.txt")

    detections = []
    if os.path.exists(label_file):
        with open(label_file, "r") as file:
            for line in file:
                data = line.strip().split()
                cls, x, y, w, h = map(float, data[0:5])

                x1 = int((x - w / 2) * frame_width)
                y1 = int((y - h / 2) * frame_height)
                x2 = int((x + w / 2) * frame_width)
                y2 = int((y + h / 2) * frame_height)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                if cv2.pointPolygonTest(road_polygon, centroid, False) >= 0:
                    detections.append([x1, y1, x2, y2, 1])  # Confidence = 1

    # Update tracker
    tracked_objects = tracker.update(np.array(detections))

    # Draw ROI polygon
    cv2.polylines(frame, [road_polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in vehicle_positions:
            vehicle_positions[track_id] = []

        vehicle_positions[track_id].append(centroid)

        if len(vehicle_positions[track_id]) > 10:
            recent_centroids = vehicle_positions[track_id][-10:]
            total_distance = sum(
                euclidean(recent_centroids[i], recent_centroids[i + 1]) for i in range(9)
            )

            # Split frame condition
            vertical_half = frame_height // 2
            if centroid[1] < vertical_half:
                threshold = 0.5 * 10  # upper half: <2 px/frame
            else:
                threshold = 1 * 10  # lower half: <1 px/frame

            if total_distance < threshold:
                speed_kmh = 0.0
            else:
                real_distance = total_distance * GSD
                speed_mps = real_distance * (FPS / 10)
                speed_kmh = speed_mps * 3.6

            vehicle_speeds[track_id] = speed_kmh
        else:
            speed_kmh = vehicle_speeds.get(track_id, 0)

        # Draw bounding box and label
        color = (0, 255, 0) if speed_kmh > 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id} | {speed_kmh:.1f} km/h", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save data
        df = pd.concat([
            df,
            pd.DataFrame([[frame_num, track_id, speed_kmh, centroid[0], centroid[1], x1, y1, x2, y2]],
                         columns=columns)
        ], ignore_index=True)

    cv2.imshow("Speed Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save results
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Speed data saved to {CSV_OUTPUT_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()
