import cv2
import pandas as pd

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\720 good.mp4"
OUTPUT_VIDEO_PATH = r"C:\Users\Sambhavi Roy\Downloads\output_with_distances.mp4"
STATIONARY_CSV_PATH = r"C:\Users\Sambhavi Roy\Downloads\stationary_data.csv"
DISTANCE_CSV_PATH = r"C:\Users\Sambhavi Roy\Downloads\frame_distances.csv"

# Load stationary vehicle data
df = pd.read_csv(STATIONARY_CSV_PATH)

# Read video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Data storage for distances
distance_records = []

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Get stationary vehicles for this frame
    frame_data = df[df["frame"] == frame_num]

    if not frame_data.empty:
        # Find max and min y2-coordinates
        max_y2 = frame_data["y2"].max()
        min_y2 = frame_data["y2"].min()

        # Calculate distance
        distance = max_y2 - min_y2
        distance_records.append([frame_num, distance])

        # Get vehicles with max and min y2-coordinates
        max_y_vehicle = frame_data[frame_data["y2"] == max_y2]
        min_y_vehicle = frame_data[frame_data["y2"] == min_y2]

        for _, row in frame_data.iterrows():
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            track_id = int(row["vehicle_id"])
            color = (255, 0, 0)  # Default: BLUE

            if track_id in max_y_vehicle["vehicle_id"].values:
                color = (0, 0, 255)  # RED (Max Y2)

            elif track_id in min_y_vehicle["vehicle_id"].values:
                color = (0, 255, 255)  # YELLOW (Min Y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay distance on frame
        cv2.putText(frame, f"Max Y2 - Min Y2 = {distance} pixels", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)  # Save frame to video
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save distances to CSV
df_distance = pd.DataFrame(distance_records, columns=["frame", "max_y2 - min_y2"])
df_distance.to_csv(DISTANCE_CSV_PATH, index=False)

print("✅ Output video saved:", OUTPUT_VIDEO_PATH)
print("✅ Distance data saved:", DISTANCE_CSV_PATH)
