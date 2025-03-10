import cv2
import pandas as pd
import numpy as np

# Constants
GSD = 0.1048  # Ground Sampling Distance (meters per pixel) at 100m altitude
ROAD_ANGLE = 40  # Road inclination in degrees
COS_ANGLE = np.cos(np.radians(ROAD_ANGLE))
MAX_GAP = 100  # Maximum pixel gap to consider a vehicle part of the queue

# Load the CSV file
csv_file = r"C:\Users\Sambhavi Roy\Downloads\stationary_data.csv"  # Update with your actual file path
df = pd.read_csv(csv_file)

# Load the video
video_path = r"C:\Users\Sambhavi Roy\Downloads\720 good.mp4"  # Update with your actual file path
cap = cv2.VideoCapture(video_path)
output_path = r"C:\Users\Sambhavi Roy\Downloads\ROIprogQueue2.mp4"

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# ORIGINAL COORDINATES (Based on manually marked image)
original_width, original_height = 1523, 1064  # Change this to match your manual marking image resolution

# Scale polygon to match video resolution
scale_x = frame_width / original_width
scale_y = frame_height / original_height

road_polygon = np.array([
    [int(50 * scale_x), int(200 * scale_y)],
    [int(150 * scale_x), int(150 * scale_y)],
    [int(1500 * scale_x), int(960 * scale_y)],
    [int(1092 * scale_x), int(1052 * scale_y)]
], np.int32)

# Create a blank mask for the AOI
mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask, [road_polygon], 255)  # Fill the AOI with white

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get stationary vehicles for the current frame
    frame_data = df[df['frame'] == frame_num]
    
    # Filter out vehicles that are not inside the road AOI
    filtered_vehicles = []
    for _, row in frame_data.iterrows():
        centroid_x = (row['x1'] + row['x2']) // 2
        centroid_y = (row['y1'] + row['y2']) // 2
        if mask[centroid_y, centroid_x] == 255:  # Only consider vehicles inside AOI
            filtered_vehicles.append(row)

    if filtered_vehicles:
        filtered_df = pd.DataFrame(filtered_vehicles)
        filtered_df = filtered_df.sort_values(by='y2', ascending=False)  # Sort vehicles from highest y2 (bottom-most) to lowest

        # Red-boxed vehicle (first in queue)
        max_vehicle = filtered_df.iloc[0]
        queue_start_y = max_vehicle['y2']
        queue_end_y = queue_start_y  # Initially, queue ends at the first vehicle

        # Extend queue if consecutive vehicles have a gap ≤ MAX_GAP
        for i in range(1, len(filtered_df)):
            curr_vehicle = filtered_df.iloc[i]
            curr_y2 = curr_vehicle['y2']

            if (queue_end_y - curr_y2) <= MAX_GAP:
                queue_end_y = curr_y2  # Extend queue
            else:
                break  # Stop at the last valid vehicle

        # Compute queue length in pixels
        queue_length_pixels = queue_start_y - queue_end_y

        # Scaling factor due to cropping of original video frames
        scaled_factor = 2.5

        # Convert to meters and project along the road
        projected_distance_m = queue_length_pixels * GSD * COS_ANGLE * scaled_factor

        # Draw bounding boxes for queue start and end vehicles
        min_vehicle = filtered_df[filtered_df['y2'] == queue_end_y].iloc[0]

        cv2.rectangle(frame, (max_vehicle['x1'], max_vehicle['y1']), (max_vehicle['x2'], max_vehicle['y2']), (0, 0, 255), 2)  # Red (Queue Start)
        cv2.rectangle(frame, (min_vehicle['x1'], min_vehicle['y1']), (min_vehicle['x2'], min_vehicle['y2']), (0, 255, 255), 2)  # Yellow (Queue End)

        # Display projected queue length in meters
        text = f"Projected Queue Length: {projected_distance_m:.2f} m"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 2)

    # Write frame to output video
    out.write(frame)
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved as", output_path)
