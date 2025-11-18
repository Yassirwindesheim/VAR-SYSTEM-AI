# ⚽ Automated Offside Detection System (YOLOv8 & Computer Vision)

## 🎯 Project Overview

This project uses **Computer Vision (CV)** and **Machine Learning (ML)** based on **YOLOv8** to automatically analyze football (soccer) frames to detect the offside status.

The system performs three primary tasks:
1.  **Object Detection:** Locates all players and the ball.
2.  **Team Assignment:** Uses **K-Means Clustering** on shirt color (LAB space) to group players into two teams.
3.  **Attack Direction:** Uses a **custom YOLOv8 Classification Model** to robustly determine which side (left/right) the attacking team is moving towards.

By combining the ML-predicted attack direction with player coordinates, the system accurately calculates the **offside line** (second-to-last defender) and flags any offside attackers.

## ✨ Key Components and Usage

| File | Purpose | Key Technology |
| :--- | :--- | :--- |
| `src/detect.py` | Runs the initial **Object Detection** (players/ball) and saves results to `data/detections.csv`. | YOLOv8 (Detection) |
| `src/logic.py` | The **Main Analysis Script**. It loads the custom classification model, performs team clustering, determines the attack direction, calculates the offside line, and generates visualizations. | YOLOv8 (Classification), K-Means, OpenCV |
| `runs/classify/yolov8_attack_direction2/weights/best.pt` | The **Custom ML Model** trained to predict the attack direction (`attacking_left` or `attacking_right`). | YOLOv8-cls |

## ⚙️ Setup and Run

1.  **Install requirements:** `pip install ultralytics pandas scikit-learn opencv-python numpy`
2.  **Run Detection:** `python src/detect.py` (Generates `detections.csv`)
3.  **Run Analysis:** `python src/logic.py` (Performs ML analysis and outputs visualization images to `outside_folder/`)

**Crucial Note:** Ensure the file path to `best.pt` in `src/logic.py` accurately points to your trained classification model.
