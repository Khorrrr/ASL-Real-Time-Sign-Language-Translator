import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path


DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "asl_alphabet_train")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

MAX_IMAGES_PER_CLASS = None  
def setup_mediapipe():
    base_options = python.BaseOptions(model_asset_path='data/processed/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def extract_landmarks_from_image(hands, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = hands.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]

    landmarks = []
    for landmark in hand_landmarks:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    return landmarks  

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    wrist_x = landmarks[0]
    wrist_y = landmarks[1]
    wrist_z = landmarks[2]

    normalized = []
    for i in range(0, len(landmarks), 3):
        normalized.append(landmarks[i] - wrist_x)
        normalized.append(landmarks[i + 1] - wrist_y)
        normalized.append(landmarks[i + 2] - wrist_z)

    max_val = max(abs(v) for v in normalized)
    if max_val > 0:
        normalized = [v / max_val for v in normalized]

    return normalized

def main():
    print("=" * 60)
    print("LANDMARK EXTRACTION")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print(f"\nERROR: Dataset directory not found!")
        print(f"Expected: {DATASET_DIR}")
 

        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        if os.path.exists(base):
            print(f"\nContents of {base}:")
            for item in os.listdir(base):
                print(f"  {item}/")
                subpath = os.path.join(base, item)
                if os.path.isdir(subpath):
                    for subitem in os.listdir(subpath)[:5]:
                        print(f"    {subitem}/")
        return

    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])

    print(f"\nFound {len(classes)} classes: {classes}")

    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    reverse_label_map = {idx: cls_name for cls_name, idx in label_map.items()}

    print(f"Label mapping: {label_map}")

    hands = setup_mediapipe()

    all_landmarks = []
    all_labels = []
    failed_count = 0
    total_count = 0

    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        image_files = [f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if MAX_IMAGES_PER_CLASS:
            image_files = image_files[:MAX_IMAGES_PER_CLASS]

        print(f"\nProcessing class '{cls_name}': {len(image_files)} images")

        cls_failed = 0
        for img_file in tqdm(image_files, desc=f"  {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_file)
            total_count += 1

            landmarks = extract_landmarks_from_image(hands, img_path)

            if landmarks is None:
                failed_count += 1
                cls_failed += 1
                continue

            normalized = normalize_landmarks(landmarks)

            all_landmarks.append(normalized)
            all_labels.append(label_map[cls_name])

        success_rate = ((len(image_files) - cls_failed) / len(image_files)) * 100
        print(f"  {cls_name}: {len(image_files) - cls_failed}/{len(image_files)} successful ({success_rate:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total images processed: {total_count}")
    print(f"Successful extractions: {len(all_landmarks)}")
    print(f"Failed (no hand found): {failed_count}")
    print(f"Success rate: {(len(all_landmarks)/total_count)*100:.1f}%")

    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")

    data = np.array(all_landmarks)
    labels = np.array(all_labels).reshape(-1, 1)
    full_data = np.hstack([data, labels])

    df = pd.DataFrame(full_data, columns=columns)
    df["label"] = df["label"].astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "landmarks.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nLandmarks saved to: {csv_path}")
    print(f"Shape: {df.shape}")

    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "label_to_idx": label_map,
            "idx_to_label": reverse_label_map
        }, f, indent=2)
    print(f"Label map saved to: {label_map_path}")

    print(f"\nClass distribution:")
    for cls_name, idx in sorted(label_map.items()):
        count = len(df[df["label"] == idx])
        print(f"  {cls_name:>10}: {count} samples")

    hands.close()

if __name__ == "__main__":
    main()