# 🚦 Queue Length, Speed Estimation & Queue Length Prediction

## 📁 Dataset

Drone footage of traffic at an intersection. Each video is 120 seconds long (red-to-red). Cycles 1–4 are combined into a single video.

---

## 📌 Project Structure

The project is divided into three parts:
1. **Vehicle Detection**
2. **Queue Length Estimation**
3. **Queue Length Prediction**

---

## 🚘 Vehicle Detection

Train YOLOv5 on the image dataset provided in the `Vehicle Detection` folder.

### 🔧 Train

```bash
python train.py --img 512 --batch 8 --epochs 15 --data path\to\data.yaml --weights yolov5s.pt --device 0 --cache=False --workers 0
```

### 📊 Evaluate

```bash
python val.py --weights path\to\weights.pt --data path\to\data.yaml --task test --img 512 --conf 0.25 --device 0
```

### 🎥 Test on Video

```bash
python detect.py --weights "path\to\weights.pt" --source "path\to\cycle9.mp4" --conf-thres 0.25 --device 0 --imgsz 416 --save-txt
```

- 🔹 Trained weights `best.pt` are also included.
- 🔹 Save the vehicle labels for all videos from 1–14.

---

## 📏 Queue Length Estimation

### 🗺 ROI

Refer to `ROI.ipynb`. Define ROI for each video individually and update the relevant code accordingly.

### 🛣 Estimating Speeds

1. Create a new environment named `tracking_vehicle` and install SORT dependencies.
2. Use `speed.py` for each video.

**Outputs:**
- Annotated video with per-frame vehicle speeds.
- CSV with columns:

```
frame, vehicle_id, speed_kmh, centroid_x, centroid_y, x1, y1, x2, y2
```

### 📐 Calculating Queue Length

Use `saveCSV.py` for each video.

**Outputs:**
- Annotated video with queue length displayed per frame.
- CSV with columns:

```
Frame, Queue_Length_m, Red_x1, Red_y1, Red_x2, Red_y2, Yellow_x1, Yellow_y1, Yellow_x2, Yellow_y2
```

- **Red**: First stationary vehicle.
- **Yellow**: Last stationary vehicle.

### 🔧 Improving Queue Values

Due to fluctuations in the yellow box (last stationary vehicle), apply post-processing:
- Once the last stationary vehicle is detected at a certain point, it should not jump ahead in later frames.
- Use `fixedQueue.py` and the CSV from the previous step.
- Then, compute **average queue length per second** for prediction.

### 🧪 Comparing with Ground Truth

Compare estimated queue lengths per second with ground truth:
- Ground truth labels are obtained by **manually measuring** the median line between the first and last stationary vehicles at each second.

---

## 🚀 Enhancements in Queue Length Estimation Methodology

### ✅ Robust Queue Definition

Initially, queue length was computed as the distance from the first to the last stationary vehicle. However, due to SORT’s occasional misclassification of faraway moving vehicles as stationary, this naive approach often **overestimated** the queue.

**Improved Method:**
- Implemented a **segmented accumulation approach** — queue length is computed by summing inter-vehicle distances from the first stationary vehicle until a significant gap is found (based on a spatial threshold). This avoids mistakenly including distant moving vehicles.

### 🔄 Temporal Smoothing Logic

Frame-by-frame queue estimates may fluctuate due to tracking inconsistencies. To ensure realism:
- During the **red phase**, the queue length can **increase or remain constant**.
- During the **green phase**, it can **decrease or remain constant**.

This enforces a **monotonic queue evolution** aligned with real-world traffic behavior.

---

## 📈 Queue Length Prediction

Use the queue lengths obtained per second for each video in the previous step.

- **Total data**: 120 × 14 = **1680 seconds**
- Use `LSTM.ipynb` to train a predictive model:

**Training:**
- Train on first **960 seconds** (Cycles 1–8)

**Testing:**
- Predict for next **720 seconds** (Cycles 9–14)
