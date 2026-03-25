"""
STEP 3: Evaluate the trained model — confusion matrix, per-class accuracy.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from train_model import ASLClassifier


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "landmarks.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letter_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)

    idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
    num_classes = len(idx_to_label)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Load scaler
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_test = scaler.transform(X_test)

    # Load model
    model = ASLClassifier(input_size=63, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "asl_model.pth"), map_location=DEVICE))
    model.eval()

    # Get predictions
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = outputs.max(1)

    predictions = predictions.cpu().numpy()

    # Class names
    class_names = [idx_to_label[i] for i in range(num_classes)]

    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(y_test, predictions, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.show()

    print("\nMost Confused Pairs:")
    print("-" * 40)

    np.fill_diagonal(cm, 0)  # Remove correct predictions
    for _ in range(10):
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        if cm[i, j] == 0:
            break
        print(f"  {class_names[i]} confused as {class_names[j]}: {cm[i, j]} times")
        cm[i, j] = 0


if __name__ == "__main__":
    main()