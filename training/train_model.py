import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "landmarks.csv")
LABEL_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "label_map.json")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letter_model")

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 10 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ASLClassifier(nn.Module):
    """Neural network for ASL letter classification."""

    def __init__(self, input_size=63, num_classes=29):
        super(ASLClassifier, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    print("Loading data.")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map["label_to_idx"])
    print(f"Number of classes: {num_classes}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes, scaler, label_map


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("ASL LETTER MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, scaler, label_map = load_data()

    # Create datasets and dataloaders
    train_dataset = LandmarkDataset(X_train, y_train)
    val_dataset = LandmarkDataset(X_val, y_val)
    test_dataset = LandmarkDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = ASLClassifier(input_size=63, num_classes=num_classes).to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "asl_model.pth"))
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "asl_model.pth")))

    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"\n{'=' * 60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    with open(os.path.join(MODEL_SAVE_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    with open(os.path.join(MODEL_SAVE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODEL_SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    config = {
        "input_size": 63,
        "num_classes": num_classes,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc
    }
    with open(os.path.join(MODEL_SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll files saved to: {MODEL_SAVE_DIR}")

    plot_training_curves(history)


def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_title("Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()