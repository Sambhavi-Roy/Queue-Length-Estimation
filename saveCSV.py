import cv2
import pandas as pd
import numpy as np

# Constants
GSD = 0.1048  # Ground Sampling Distance (meters per pixel) at 100m altitude
ROAD_ANGLE = 40  # Road inclination in degrees
COS_ANGLE = np.cos(np.radians(ROAD_ANGLE))
MAX_GAP = 50  # Maximum pixel gap to consider a vehicle part of the queue

# Load the CSV file
csv_file = r"C:\Users\Sambhavi Roy\Downloads\sample 9\9speed_data.csv"
df = pd.read_csv(csv_file)
df = df[df['speed_kmh'] == 0]

# Load the video
video_path = r"C:\Users\Sambhavi Roy\Downloads\sample 9\video 9.mp4"
cap = cv2.VideoCapture(video_path)
output_path = r"C:\Users\Sambhavi Roy\Downloads\sample 9\SavedspeedROIprogQueue.mp4"

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# ORIGINAL COORDINATES
original_width, original_height = 1540, 1079
scale_x = frame_width / original_width
scale_y = frame_height / original_height


# Define road polygon points in reference resolution
road_polygon = np.array([
    [200, 206],
    [274, 168],
    [1512, 972],
    [1254, 1060],
], np.int32)


mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask, [road_polygon], 255)

# CSV file to store results
results = []
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_data = df[df['frame'] == frame_num]
    filtered_vehicles = []
    for _, row in frame_data.iterrows():
        centroid_x = int((row['x1'] + row['x2']) // 2)
        centroid_y = int((row['y1'] + row['y2']) // 2)
        if mask[centroid_y, centroid_x] == 255:
            filtered_vehicles.append(row)

    if filtered_vehicles:
        filtered_df = pd.DataFrame(filtered_vehicles).sort_values(by='y2', ascending=False)
        max_vehicle = filtered_df.iloc[0]
        queue_start_y = max_vehicle['y2']
        queue_end_y = queue_start_y

        for i in range(1, len(filtered_df)):
            curr_vehicle = filtered_df.iloc[i]
            curr_y2 = curr_vehicle['y2']
            if (queue_end_y - curr_y2) <= MAX_GAP:
                queue_end_y = curr_y2
            else:
                break

        queue_length_pixels = queue_start_y - queue_end_y
        scaled_factor = 2.48
        projected_distance_m = queue_length_pixels * GSD * COS_ANGLE * scaled_factor

        min_vehicle = filtered_df[filtered_df['y2'] == queue_end_y].iloc[0]
        cv2.rectangle(frame, (int(max_vehicle['x1']), int(max_vehicle['y1'])), (int(max_vehicle['x2']), int(max_vehicle['y2'])), (0, 0, 255), 2)
        cv2.rectangle(frame, (int(min_vehicle['x1']), int(min_vehicle['y1'])), (int(min_vehicle['x2']), int(min_vehicle['y2'])), (0, 255, 255), 2)
        
        text = f"Projected Queue Length: {projected_distance_m:.2f} m"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 2)
        
        results.append([frame_num, projected_distance_m, max_vehicle['x1'], max_vehicle['y1'], max_vehicle['x2'], max_vehicle['y2'], min_vehicle['x1'], min_vehicle['y1'], min_vehicle['x2'], min_vehicle['y2']])
    
    out.write(frame)
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save results to CSV
results_df = pd.DataFrame(results, columns=['Frame', 'Queue_Length_m', 'Red_x1', 'Red_y1', 'Red_x2', 'Red_y2', 'Yellow_x1', 'Yellow_y1', 'Yellow_x2', 'Yellow_y2'])
results_df.to_csv(r"C:\Users\Sambhavi Roy\Downloads\sample 9\9queue_data.csv", index=False)

print("Video processing complete. Output saved as", output_path)
print("Queue data saved as queue_data.csv")
