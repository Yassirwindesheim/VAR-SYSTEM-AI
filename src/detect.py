from ultralytics import YOLO
import csv
import glob
from pathlib import Path

# --- Always use absolute paths ---
ROOT = Path(__file__).resolve().parent.parent  # .../offside-ai-full
DATA_DIR = ROOT / "data"
OUT_CSV = DATA_DIR / "detections.csv"

print(f"Saving detections to: {OUT_CSV}")

# Load model
model = YOLO("yolo11s.pt")

image_paths = []

image_paths.extend(glob.glob(str(DATA_DIR / "*.jpg")))
# Find all .png files
image_paths.extend(glob.glob(str(DATA_DIR / "*.png")))

# Run detection (classes 0 = person, 32 = ball)
results = model(image_paths, conf=0.03, imgsz=1280, classes=[0, 32])

rows_written = 0

# --- Write detections ---
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # ✅ Correct header order
    writer.writerow([
        "image_name", "class",
        "x1", "y1", "x2", "y2",
        "x_center", "y_center",
        "conf"
    ])

    # Loop through images and their detections
    for path, result in zip(image_paths, results):
        image_name = Path(path).name

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # ✅ Values now match header order
            writer.writerow([
                image_name, cls,
                round(x1, 2), round(y1, 2),
                round(x2, 2), round(y2, 2),
                round(x_center, 2), round(y_center, 2),
                conf
            ])
         

print(f"✅ Done! Wrote  detections to:\n{OUT_CSV}")
