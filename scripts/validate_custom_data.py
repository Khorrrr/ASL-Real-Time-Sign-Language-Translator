import os
import numpy as np

DATA_PATH = os.path.join("data", "dynamic", "custom")
WORDS = ["hello", "help", "thank you", "yes", "no"]

def validate_data():
    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} does not exist.")
        return

    total_samples = 0
    bad_samples = []

    print(f"{'='*50}")
    print("DATA VALIDATION REPORT")
    print(f"{'='*50}")

    for word in WORDS:
        word_dir = os.path.join(DATA_PATH, word)
        if not os.path.exists(word_dir):
            print(f"Warning: Directory for word '{word}' not found.")
            continue

        files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
        print(f"\nWord: {word.upper()} ({len(files)} samples found)")
        
        for file in files:
            total_samples += 1
            file_path = os.path.join(word_dir, file)
            
            try:
                data = np.load(file_path)
            except Exception as e:
                print(f"  [ERROR] {file}: Failed to load file.")
                bad_samples.append(file_path)
                continue
            
            if data.shape != (30, 126):
                print(f"  [WARNING] {file}: Incorrect shape {data.shape}. Expected (30, 126).")
                bad_samples.append(file_path)
                continue
            
            empty_frames = 0
            for frame in data:
                if not np.any(frame):
                    empty_frames += 1
            
            if empty_frames > 15:
                print(f"  [BAD] {file}: {empty_frames}/30 frames have NO hands detected!")
                bad_samples.append(file_path)
            elif empty_frames > 5:
                print(f"  [WARN] {file}: {empty_frames}/30 frames have no hands detected.")
                
    print(f"\n{'='*50}")
    print(f"Total Samples Checked: {total_samples}")
    if bad_samples:
        print(f"Total Bad/Suspicious Samples: {len(bad_samples)}")
        print("\nACTION REQUIRED: Please delete the following files and re-record them:")
        for bad in bad_samples:
            print(f" - {bad}")
    else:
        print("You are ready to train the model.")

if __name__ == "__main__":
    validate_data()
