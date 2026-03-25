import cv2
import math
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# --- APP STATE & TRACKING VARIABLES ---
app_state = "IDLE"
calories_burned = 0.0

muscle_load = {
    "Arms (Biceps/Triceps)": 0.0,
    "Shoulders/Chest/Back": 0.0,
    "Core/Glutes": 0.0,
    "Legs (Quads/Hams)": 0.0
}

smoothed_angles = {}
FILTER_ALPHA = 0.4  # EMA Filter: 0.1 is very slow/smooth, 0.9 is fast/jittery
is_gripping = False

# --- MOUSE CLICK HANDLER ---
def handle_click(event, x, y, flags, param):
    global app_state, calories_burned, muscle_load, smoothed_angles
    if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 160 and 10 <= y <= 60:
        if app_state == "IDLE":
            app_state = "ACTIVE"
            calories_burned = 0.0
            muscle_load = {k: 0.0 for k in muscle_load}
            smoothed_angles = {}
        elif app_state == "ACTIVE":
            app_state = "SUMMARY"
        elif app_state == "SUMMARY":
            app_state = "IDLE"

# --- HELPER FUNCTIONS ---
def calc_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    return 360 - angle if angle > 180.0 else angle

def check_grip(x, y, box_x1, box_y1, box_x2, box_y2):
    buffer = 60 
    return (box_x1 - buffer < x < box_x2 + buffer) and (box_y1 - buffer < y < box_y2 + buffer)

# --- MODEL INITIALIZATION ---
yolo_model = YOLO('yolov8n.pt') 

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
detector = vision.PoseLandmarker.create_from_options(options)

# --- OPENCV SETUP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('StretchMaster')
cv2.setMouseCallback('StretchMaster', handle_click)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    height, width, _ = frame.shape
    found_grip_this_frame = False
    
    if app_state in ["IDLE", "ACTIVE"]:
        # 1. RUN YOLO
        yolo_results = yolo_model(frame, stream=True, verbose=False) 
        object_boxes = []
        for r in yolo_results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    object_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'WEIGHT', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. RUN MEDIAPIPE
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_results = detector.detect(mp_image)
        
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks[0]
            
            left_wrist_x, left_wrist_y = int(lm[15].x * width), int(lm[15].y * height)
            right_wrist_x, right_wrist_y = int(lm[16].x * width), int(lm[16].y * height)
            
            # SENSOR FUSION: Check if either wrist is gripping a weight
            for box in object_boxes:
                if check_grip(left_wrist_x, left_wrist_y, *box) or check_grip(right_wrist_x, right_wrist_y, *box):
                    found_grip_this_frame = True
                    break
            
            is_gripping = found_grip_this_frame

            # Draw wrist trackers
            dot_color = (0, 255, 0) if is_gripping else (0, 0, 255)
            cv2.circle(frame, (left_wrist_x, left_wrist_y), 10, dot_color, -1)
            cv2.circle(frame, (right_wrist_x, right_wrist_y), 10, dot_color, -1)

            # --- FULL BODY KINEMATICS WITH EMA FILTER ---
            raw_angles = {
                "Arms (Biceps/Triceps)": (calc_angle(lm[11], lm[13], lm[15]) + calc_angle(lm[12], lm[14], lm[16])) / 2,
                "Shoulders/Chest/Back": (calc_angle(lm[23], lm[11], lm[13]) + calc_angle(lm[24], lm[12], lm[14])) / 2,
                "Core/Glutes": (calc_angle(lm[11], lm[23], lm[25]) + calc_angle(lm[12], lm[24], lm[26])) / 2,
                "Legs (Quads/Hams)": (calc_angle(lm[23], lm[25], lm[27]) + calc_angle(lm[24], lm[26], lm[28])) / 2
            }
            
            if app_state == "ACTIVE":
                total_movement_this_frame = 0
                
                # Dynamic multiplier: Upper body works harder if gripping a weight
                weight_multiplier = 2.0 if is_gripping else 1.0

                for group in raw_angles:
                    if group not in smoothed_angles:
                        smoothed_angles[group] = raw_angles[group] # Initialize first frame
                    else:
                        # Apply the EMA Smoothing Filter
                        prev_angle = smoothed_angles[group]
                        new_angle = (FILTER_ALPHA * raw_angles[group]) + ((1 - FILTER_ALPHA) * prev_angle)
                        smoothed_angles[group] = new_angle
                        
                        # Calculate Delta from smoothed data
                        delta = abs(new_angle - prev_angle)
                        
                        if delta > 1.5: # Deadzone to ignore micro-jitters
                            # Apply multiplier only to upper body groups if holding weight
                            multiplier = weight_multiplier if "Arms" in group or "Shoulders" in group else 1.0
                            
                            muscle_load[group] += (delta * multiplier)
                            total_movement_this_frame += (delta * multiplier)
                            
                calories_burned += (total_movement_this_frame * 0.0005)

    # --- USER INTERFACE ---
    btn_color = (0, 255, 0) if app_state == "IDLE" else (0, 0, 255) if app_state == "ACTIVE" else (255, 165, 0)
    btn_text = "START" if app_state == "IDLE" else "STOP" if app_state == "ACTIVE" else "CLOSE"
    cv2.rectangle(frame, (10, 10), (160, 60), btn_color, -1)
    cv2.putText(frame, btn_text, (35, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Show WEIGHT GRIPPED status
    if is_gripping and app_state == "ACTIVE":
        cv2.putText(frame, 'WEIGHT DETECTED: 2x EXERTION', (180, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if app_state == "ACTIVE":
        cv2.rectangle(frame, (10, 80), (320, 260), (0, 0, 0), -1)
        cv2.putText(frame, f'Kcals: {calories_burned:.2f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, 'Muscle Load (Filtered):', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset = 180
        for group, load in muscle_load.items():
            cv2.putText(frame, f'{group}: {int(load)}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

    if app_state == "SUMMARY":
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 80), (width - 50, height - 50), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        
        cv2.putText(frame, 'FULL BODY SUMMARY', (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f'Total Calories: {calories_burned:.2f} kcal', (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        most_exerted = max(muscle_load, key=muscle_load.get)
        
        cv2.putText(frame, 'PRIMARY FATIGUE DETECTED:', (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, most_exerted.upper(), (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(frame, 'RECOMMENDED STRETCHES:', (70, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if most_exerted == "Arms (Biceps/Triceps)":
            stretches = ["1. Wall Bicep Stretch (60s)", "2. Overhead Tricep Stretch (60s)"]
        elif most_exerted == "Shoulders/Chest/Back":
            stretches = ["1. Doorway Pec Stretch (60s)", "2. Cross-Body Shoulder Stretch (60s)"]
        elif most_exerted == "Core/Glutes":
            stretches = ["1. Cobra Pose (Abdominals) (60s)", "2. Supine Glute Stretch (60s)"]
        else:
            stretches = ["1. Standing Quad Stretch (60s)", "2. Seated Hamstring Reach (60s)"]
            
        cv2.putText(frame, stretches[0], (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, stretches[1], (70, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('StretchMaster', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()