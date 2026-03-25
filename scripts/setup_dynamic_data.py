import os
import json
import requests
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


TARGET_WORDS = ["hello", "help", "thank you", "yes", "no"]
METADATA_URL = "https://raw.githubusercontent.com/dxli522/WLASL/master/start_kit/WLASL_v0.3.json"
DATA_DIR = "data/dynamic"
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")

SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def download_metadata():
    """Downloads the WLASL mapping file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    json_path = os.path.join(DATA_DIR, "WLASL_v0.3.json")
    if not os.path.exists(json_path):
        print(f"Downloading metadata from {METADATA_URL}...")
        r = requests.get(METADATA_URL)
        with open(json_path, 'wb') as f:
            f.write(r.content)
    return json_path

def get_video_ids(json_path):
    """Filters metadata for target words and returns a list of video IDs."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    target_data = []
    for entry in data:
        if entry['gloss'] in TARGET_WORDS:
            for video in entry['instances']:
                target_data.append({
                    "word": entry['gloss'],
                    "video_id": video['video_id'],
                    "url": video['url']
                })
    
    df = pd.DataFrame(target_data)
    df.to_csv(os.path.join(DATA_DIR, "target_videos.csv"), index=False)
    print(f"Found {len(df)} video instances for words: {TARGET_WORDS}")
    print(df.groupby('word').size())
    return df

def extract_landmarks_from_video(video_path):
    """Process a single video and returns a sequence of landmarks."""
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        frame_landmarks = np.zeros(63) 
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            for i, lm in enumerate(landmarks.landmark):
                frame_landmarks[i*3] = lm.x
                frame_landmarks[i*3+1] = lm.y
                frame_landmarks[i*3+2] = lm.z
        
        sequence.append(frame_landmarks)
    
    cap.release()
    
    if len(sequence) < SEQUENCE_LENGTH:
        padding = [sequence[-1]] * (SEQUENCE_LENGTH - len(sequence)) if sequence else [np.zeros(63)] * SEQUENCE_LENGTH
        sequence.extend(padding)
    else:
        start = (len(sequence) - SEQUENCE_LENGTH) // 2
        sequence = sequence[start : start + SEQUENCE_LENGTH]
        
    return np.array(sequence)

def main():
    json_path = download_metadata()
    df = get_video_ids(json_path)

if __name__ == "__main__":
    main()
