import pandas as pd
from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
from ultralytics import YOLO 
from pathlib import Path    


# --- Setup Constants for Drawing ---
COLOR_TEAM_0 = (255, 0, 0)  # Blue
COLOR_TEAM_1 = (0, 0, 255)  # Red
COLOR_BALL = (0, 255, 255)  # Yellow
COLOR_OFFSIDE_LINE = (0, 255, 0)  # Green 
OUTPUT_FOLDER = "../outside_folder"


# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# -------------------------------------------------------------
# 🤖 CLASSIFICATION MODEL SETUP
# -------------------------------------------------------------


ROOT = Path(__file__).resolve().parent.parent 


CLASSIFICATION_MODEL_PATH = r'C:\Users\yassi\runs\classify\yolov8_attack_direction2\weights\best.pt'

CLASSIFICATION_KLASSEN = ['attacking_left', 'attacking_right'] 

# Laad het classificatie model slechts één keer
try:
    model_direction = YOLO(str(CLASSIFICATION_MODEL_PATH))
    print(f"✅ Classificatiemodel geladen: {CLASSIFICATION_MODEL_PATH}")
except Exception as e:
    print(f"❌ Kon classificatiemodel niet laden: {e}. Controleer het pad!")
    exit()

def voorspel_aanvalsrichting(frame: np.ndarray) -> str:
    """Voorspelt de aanvalsrichting ('attacking_left' of 'attacking_right') met ML."""
    
    results = model_direction.predict(
        source=frame,
        imgsz=640,
        verbose=False 
    )
    
    # Zoek de index met de hoogste waarschijnlijkheid
    top_index = results[0].probs.top1 
    return CLASSIFICATION_KLASSEN[top_index]

# -------------------------------------------------------------
# --- Main Logica ---
data = pd.read_csv("../data/detections.csv")
image_names = data["image_name"].unique()


for image in image_names:
    frame_data = data[data["image_name"] == image]
    players = frame_data[frame_data["class"] == 0]
    # Bal is nog steeds gedetecteerd, maar we maken de check optioneel
    ball = frame_data[frame_data["class"] == 32]


    img_path = f"../data/{image}"
    frame_img = cv2.imread(img_path)


    if frame_img is None:
        print(f"⚠️ Couldn't load {img_path}")
        continue


    if len(players) < 2:
        print(f"{image}: not enough players for team detection.")
        continue

    h, w = frame_img.shape[:2]
    
    # Initialize feature lists
    features = [] 
    player_indices = []


   
    for idx, row in players.iterrows():
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        
        # Bounding box safety checks
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        
        # Crop top 35% for shirt region
        shirt_height = int((y2 - y1) * 0.35)
        top_y2 = y1 + shirt_height
        crop = frame_img[y1:top_y2, x1:x2]
        
        if crop.size == 0:
            continue
        
      
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        
        # Calculate mean LAB values
        mean_l = np.mean(lab[:, :, 0])  # Lightness
        mean_a = np.mean(lab[:, :, 1])  # Green-Red axis
        mean_b = np.mean(lab[:, :, 2])  # Blue-Yellow axis
        
        features.append([mean_l, mean_a, mean_b])
        player_indices.append(idx)
    
    # Check if enough valid crops
    if len(features) < 2:
        print(f"{image}: not enough valid crops for K-Means.")
        continue


    # --- K-Means Clustering voor Team  ---
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = kmeans.fit_predict(features)


    players = players.copy()
    players.loc[player_indices, "team"] = labels
    players["team"] = players["team"].astype(int)


    # --- Bepaal welke team-index 0 of 1 de aanvallers zijn ---
    
    # Bepaal de gemiddelde X-positie (NODIG om K-Means labels aan ML-richting te koppelen)
    team_0_avg_x = players[players["team"] == 0]["x_center"].mean()
    team_1_avg_x = players[players["team"] == 1]["x_center"].mean()


    # 1. Voorspel de richting met het getrainde model
    attacking_direction_ml = voorspel_aanvalsrichting(frame_img)
    print(f"ML Voorspelling: {image} -> {attacking_direction_ml}")
    
    # 2. Koppel de teamnummers (0 en 1) aan de ML-voorspelling
    
    if attacking_direction_ml == "attacking_right":
        if team_0_avg_x < team_1_avg_x:
            attacking_team = 0
            defending_team = 1
        else:
            attacking_team = 1
            defending_team = 0
        
    elif attacking_direction_ml == "attacking_left":
        if team_0_avg_x > team_1_avg_x:
            attacking_team = 0
            defending_team = 1
        else:
            attacking_team = 1
            defending_team = 0
    else:
        print(f"❌ Onbekende richting: {attacking_direction_ml}. Frame overgeslagen.")
        continue

    # De string 'right' of 'left' is nodig voor de offside check logica
    direction_string = "right" if 'right' in attacking_direction_ml else "left"
    print(f"{image}: Team {attacking_team} attacks {direction_string}")


    defenders = players[players["team"] == defending_team]
    attackers = players[players["team"] == attacking_team]
    defenders_x = defenders["x_center"]


    # --- Select Offside Line (Second-to-Last Defender) ---
    offside_line = None
    if len(defenders_x) >= 2:
        if 'right' in attacking_direction_ml:  # Right attack
            offside_line = defenders_x.nsmallest(2).iloc[-1]
        else:  # Left attack
            offside_line = defenders_x.nlargest(2).iloc[-1]
    elif len(defenders_x) == 1:
        offside_line = defenders_x.iloc[0]
        print(f"{image}: Only 1 defender detected. Offside line set to this defender's position.")
    else:
        print(f"{image}: No defenders detected. Skipping offside check and viz.")
        # Dit is het eerste punt waarop we de continue verwijderen, we hebben alleen de offside_line niet
        offside_line = 0 if 'right' in attacking_direction_ml else w # Gebruik de zijlijn als fallback (of gewoon verdergaan)
    
    
    # -----------------------------------------------------------------
    #  BAL CHECK EN POSITIE BEPALING (NU OPTIONEEL)
    # -----------------------------------------------------------------
    ball_detected = len(ball) > 0
    
    if ball_detected:
        ball_x = float(ball["x_center"].iloc[0])
        ball_y = float(ball["y_center"].iloc[0])
        print(f"🖼 {image} | Ball X={ball_x:.2f} | Offside Line X={offside_line:.2f}")
    else:
        # Als de bal ontbreekt, kunnen we de offside check niet uitvoeren
        print(f"⚠️ {image}: No ball detected. Offside check skipped.")
        ball_x, ball_y = None, None # Stel de balposities in op None


    
    # -----------------------------------------------------------------
    # 🎨 VISUALIZATION LOGIC
    # -----------------------------------------------------------------
    display_img = frame_img.copy() 
    H, W = frame_img.shape[:2] 


    # 1. Draw Offside Line (Alleen als deze getrokken kon worden)
    if offside_line is not None:
        line_x_int = int(offside_line)
        cv2.line(display_img, (line_x_int, 0), (line_x_int, H), COLOR_OFFSIDE_LINE, 3) 
        cv2.putText(display_img, "OFFSIDE LINE", (line_x_int + 5, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_OFFSIDE_LINE, 2)


    # 2. Draw Player Bounding Boxes and Labels
    for _, p in players.iterrows():
        x1, y1, x2, y2 = map(int, [p["x1"], p["y1"], p["x2"], p["y2"]])
        is_attacker = (p["team"] == attacking_team)
        
        team_color = COLOR_TEAM_0 if p["team"] == 0 else COLOR_TEAM_1
        
        # Draw rectangle
        cv2.rectangle(display_img, (x1, y1), (x2, y2), team_color, 2)
        
        # Check and highlight offside attackers (ALLEEN ALS BAL EN OFFSIDE LINE BEKEND ZIJN)
        if is_attacker and ball_detected and offside_line is not None:
            player_x = p["x_center"]
            # Offside Check: ahead of ball AND ahead of offside line
            is_ahead_ball = (player_x > ball_x) if 'right' in attacking_direction_ml else (player_x < ball_x)
            is_ahead_offside = (player_x > offside_line) if 'right' in attacking_direction_ml else (player_x < offside_line)
            
            if is_ahead_ball and is_ahead_offside:
                # Highlight offside players
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(display_img, "OFFSIDE", (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        # Draw team label
        cv2.putText(display_img, f"Team {p['team']}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)


    # 3. Draw the Ball (ALLEEN ALS BAL GEVONDEN IS)
    if ball_detected:
        cv2.circle(display_img, (int(ball_x), int(ball_y)), 8, COLOR_BALL, -1) 


    # 4. Save the Result Image
    viz_filename = os.path.join(OUTPUT_FOLDER, f"offside_viz_{image}")
    cv2.imwrite(viz_filename, display_img)
    print(f"✅ Viz saved to {viz_filename}")


    # -----------------------------------------------------------------
    # 📊 TEKST OFFSIDE CHECK LOGICA
    # -----------------------------------------------------------------
    
    if not ball_detected:
        print(f"⚠️ {image}: Kan buitenspelstatus niet bepalen, bal ontbreekt.\n")
        continue

    # Als de bal is gedetecteerd, voer de buitenspelcheck uit
    offside_players = []
    
    for idx, attacker in attackers.iterrows():
        player_x = attacker["x_center"]


        if 'right' in attacking_direction_ml: 
            is_ahead_ball = player_x > ball_x
            is_ahead_offside = player_x > offside_line
        else:  # left
            is_ahead_ball = player_x < ball_x
            is_ahead_offside = player_x < offside_line


        if is_ahead_ball and is_ahead_offside:
            offside_players.append(f"Player {idx} (x={player_x:.2f})")
            
    if offside_players:
        print(f"🚨 {image}: OFFSIDE detected! Players: {', '.join(offside_players)}\n")
    else:
        print(f"✅ {image}: No offside detected.\n")
