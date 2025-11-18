# training.py (GECORRIGEERDE VERSIE)

from ultralytics import YOLO
from pathlib import Path  # Importeer Path om paden correct te hanteren

# --- Configuratie ---
MODEL_NAAM = 'yolov8_attack_direction'

# RELATIEF PAD naar de HOOFDMAP van de dataset.

# Dit pad is RELATIEF t.o.v. de 'offside-ai-full' map.
DATASET_ROOT_PATH = Path('../data/My First Project.v1-datasetfootball.folder') 

YOLO_DATA_PATH = str(DATASET_ROOT_PATH.resolve())


if not Path(YOLO_DATA_PATH).exists():
    raise FileNotFoundError(f"Fout: Dataset map niet gevonden op {YOLO_DATA_PATH}")

#  Model laden (Classification)
print("Model laden: yolov8n-cls.pt")
model = YOLO('yolov8n-cls.pt') 

#  Training uitvoeren
print("Training van Classification Model gestart...")
    
results = model.train(
    # Geef het absolute pad naar de root-map van de dataset
    data=YOLO_DATA_PATH,  
    epochs=100,           
    imgsz=640,            
    name=MODEL_NAAM,
    task='classify' 
)
    
print("\n--- Training voltooid ---")
