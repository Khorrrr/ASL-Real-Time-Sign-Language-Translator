import cv2
import mediapipe as mp
import numpy as np
import os
import time

WORDS = ["hello", "help", "thank you", "yes", "no"]
DATA_PATH = os.path.join("data", "dynamic", "custom")
SEQUENCE_LENGTH = 30  
NUM_SAMPLES = 20      

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def setup_folders():
    for word in WORDS:
        path = os.path.join(DATA_PATH, word)
        if not os.path.exists(path):
            os.makedirs(path)

def get_landmarks_array(results):
    """
    Extracts 126 landmarks (63 for Hand 1, 63 for Hand 2).
    """
    full_landmarks = np.zeros(126)
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i >= 2: break
            start_idx = i * 63
            coords = []
            for lm in hand_lms.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            full_landmarks[start_idx : start_idx + 63] = coords
    return full_landmarks

def record_word(word_index):
    word = WORDS[word_index]
    cap = cv2.VideoCapture(0)
    
    word_dir = os.path.join(DATA_PATH, word)
    existing_files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
    sample_num = len(existing_files)
    
    print(f"\n--- RECORDING FOR: {word.upper()} ---")
    print("1. Press 'R' to start recording (1s countdown).")
    print("2. Press 'N' for Next Word, 'Q' to Quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"MODE: DATA COLLECTION | WORD: {word.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples Collected: {sample_num}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "[R] Record  [N] Next Word  [Q] Quit", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("ASL Dynamic Data Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            for countdown in range(1, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"GET READY: {countdown}", (180, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                cv2.imshow("ASL Dynamic Data Recorder", frame)
                cv2.waitKey(1000)

            sequence = []
            print(f"Recording sample {sample_num}...")
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                frame_lms = get_landmarks_array(results)
                sequence.append(frame_lms)
                
                cv2.rectangle(frame, (0,0), (640, 40), (0, 0, 255), -1)
                cv2.putText(frame, f"CAPTURING FRAME: {frame_num}/{SEQUENCE_LENGTH}", (150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow("ASL Dynamic Data Recorder", frame)
                cv2.waitKey(1)

            file_path = os.path.join(DATA_PATH, word, f"{sample_num}.npy")
            np.save(file_path, np.array(sequence))
            sample_num += 1
            print(f"Success: {file_path}")

        elif key == ord('n'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return "quit"

    cap.release()
    cv2.destroyAllWindows()
    return "next"

if __name__ == "__main__":
    setup_folders()
    for i in range(len(WORDS)):
        status = record_word(i)
        if status == "quit":
            break
    print("\nData collection complete!")
