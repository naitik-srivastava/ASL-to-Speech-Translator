from ultralytics import YOLO
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
from gtts import gTTS
import tempfile
import os
import pygame
import time

def speak(text):
    if not text.strip():
        return

    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        tts = gTTS(text=text, lang='en')
        tts.save(path)

        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(1.0)  # set volume to max
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        time.sleep(0.2)
    finally:
        for _ in range(5):
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            print(f"Warning: Could not delete temp file {path}")






# Initialize models
model = YOLO('models\best.pt')
detector = HandDetector(maxHands=2, detectionCon=0.7)
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Keep this enabled
    min_detection_confidence=0.7,  # Increased from 0.5
    min_tracking_confidence=0.7)   # Increased from 0.5

hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)

# Configuration
input_size = 224
cooldown = 0.8
sentence = ""
last_registered_time = 0



# Create windows
cv2.namedWindow("ASL Translator", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left Hand Input", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Hand Input", input_size, input_size)

# Font settings
font_scale_sentence = 0.8
font_scale_name = 0.45
font_thickness = 1
font_face = cv2.FONT_HERSHEY_SIMPLEX

# Color settings
color_sentence = (10, 255, 255)  # Vibrant yellow
color_name = (255, 128, 0)       # Orange
color_name_outline = (0, 0, 0)    # Black
color_left_indicator = (0, 255, 0)  # Green
color_right_indicator = (0, 0, 255) # Red

# Custom white mesh style
mesh_style = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # Pure white
    thickness=1,            # Fine stroke
    circle_radius=0         # No dots
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    # Initialize displays
    left_hand_display = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    display_frame = frame.copy()
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

     # Face processing
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw white mesh
            mp_drawing.draw_landmarks(
                image=display_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 255, 0), 
                    thickness=1)
            )

    # Hand detection and classification
    hands = detector.findHands(frame, draw=False)
    left_gesture = None
    right_visible = False

    if hands:
        height, width, _ = frame.shape
        for hand in hands:
            x, y, w, h = hand['bbox']
            
            # Improved bounding box
            scale_factor = 2.0
            min_crop_size = 180
            size = max(int(max(w, h) * scale_factor), min_crop_size)
            
            cx, cy = x + w//2, y + h//2
            x1, y1 = max(0, cx-size//2), max(0, cy-size//2)
            x2, y2 = min(width, cx+size//2), min(height, cy+size//2)
            
            # Crop hand region
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0:
                continue
                
            # Correct hand type (accounts for mirror flip)
            actual_hand_type = 'Left' if hand['type'] == 'Right' else 'Right'
            
            if actual_hand_type == 'Left':
                # Process left hand for classification
                hand_img = cv2.resize(hand_crop, (input_size, input_size))
                left_hand_display = hand_img.copy()
                hand_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                
                # Classify left hand gesture
                results = model(hand_rgb)
                class_name = results[0].names[results[0].probs.top1]
                confidence = float(results[0].probs.top1conf)
                left_gesture = class_name
                
                # Visual feedback
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Left: {class_name} ({confidence:.2f})", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Right hand detection only
                right_visible = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, "Right: Register", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # MediaPipe hand joints
    results_mp = hands_mp.process(rgb_frame)
    if results_mp.multi_hand_landmarks:
        for hand_landmarks in results_mp.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(220, 0, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200, 255, 0), thickness=1, circle_radius=1))
    
    # Registration logic
    if right_visible and left_gesture and (current_time - last_registered_time) > cooldown:
        if left_gesture == 'del':
            sentence = sentence[:-1]
        elif left_gesture == 'space':
            sentence += ' '
        elif left_gesture not in ['nothing', 'del', 'space']:
            sentence += left_gesture
        
        last_registered_time = current_time
    
    # Display outputs
    cv2.putText(display_frame, f"Translation: {sentence}", (20, 50), 
               font_face, font_scale_sentence, color_sentence, 2)
    
    text_name = "ASL TRANSLATOR BY: NAITIK SRIVASTAVA"
    text_size = cv2.getTextSize(text_name, font_face, font_scale_name, font_thickness)[0]
    text_x = 10
    text_y = display_frame.shape[0] - 10
    
    cv2.putText(display_frame, text_name, (text_x, text_y), font_face, font_scale_name, 
               color_name_outline, font_thickness + 2, cv2.LINE_AA)
    cv2.putText(display_frame, text_name, (text_x, text_y), font_face, font_scale_name, 
               color_name, font_thickness, cv2.LINE_AA)
    
    # Visual hand indicators
    cv2.putText(display_frame, "L", (50, display_frame.shape[0]-50), 
               font_face, 1, color_left_indicator, font_thickness)
    cv2.putText(display_frame, "R", (display_frame.shape[1]-50, display_frame.shape[0]-50), 
               font_face, 1, color_right_indicator, font_thickness)
    
    # Show windows
    cv2.imshow("ASL Translator", display_frame)
    cv2.imshow("Left Hand Input", left_hand_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
     break
    elif key == ord('t'):
      speak(sentence)


cap.release()
hands_mp.close()
face_mesh.close()
cv2.destroyAllWindows()
