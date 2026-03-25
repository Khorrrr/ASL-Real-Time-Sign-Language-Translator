"""
Download the MediaPipe Hand Landmarker model.
Required for backend/hand_tracker.py to work.
"""

import urllib.request
import os

def download_model():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    save_path = os.path.join(save_dir, "hand_landmarker.task")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading hand_landmarker.task to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Done! You can now run the backend.")

if __name__ == "__main__":
    download_model()