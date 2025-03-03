import cv2
import pandas as pd
import numpy as np

# Constants
GSD = 0.1048  # Ground Sampling Distance (meters per pixel) at 100m altitude
ROAD_ANGLE = 40  # Road inclination in degrees
COS_ANGLE = np.cos(np.radians(ROAD_ANGLE))

# Load the CSV file
csv_file = r"C:\Users\Sambhavi Roy\Downloads\stationary_data.csv"  # Update with your actual file path
df = pd.read_csv(csv_file)

# Load the video
video_path = r"C:\Users\Sambhavi Roy\Downloads\720 good.mp4"  # Update with your actual file path
cap = cv2.VideoCapture(video_path)
output_path = r"C:\Users\Sambhavi Roy\Downloads\projdist2.mp4"

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get stationary vehicles for the current frame
    frame_data = df[df['frame'] == frame_num]
    
    if not frame_data.empty:
        max_y2 = frame_data['y2'].max()
        min_y2 = frame_data['y2'].min()
        
        # Compute pixel distance
        pixel_distance = max_y2 - min_y2
        
        #scaling facotr due to cropping of original video frames
        scaled_factor = 2.5

        # Convert to meters and project along the road
        projected_distance_m = pixel_distance * GSD * COS_ANGLE * scaled_factor
        
        # Draw bounding boxes for max and min y2 vehicles
        max_vehicle = frame_data[frame_data['y2'] == max_y2].iloc[0]
        min_vehicle = frame_data[frame_data['y2'] == min_y2].iloc[0]
        
        cv2.rectangle(frame, (max_vehicle['x1'], max_vehicle['y1']), (max_vehicle['x2'], max_vehicle['y2']), (0, 0, 255), 2)  # Red
        cv2.rectangle(frame, (min_vehicle['x1'], min_vehicle['y1']), (min_vehicle['x2'], min_vehicle['y2']), (0, 255, 255), 2)  # Yellow
        
        # Display projected distance in meters
        text = f"Projected Queue Length: {projected_distance_m:.2f} m"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 2)

    
    # Write frame to output video
    out.write(frame)
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved as", output_path)
