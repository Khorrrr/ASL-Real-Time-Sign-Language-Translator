import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import random


DATA_PATH = os.path.join("data", "dynamic", "custom")
MODEL_DIR = os.path.join("models", "dynamic_model")
WORDS = ["hello", "help", "thank you", "yes", "no"]

SEQUENCE_LENGTH = 30
INPUT_SIZE = 126 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = len(WORDS)

BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.0005 
WEIGHT_DECAY = 1e-4    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PREPROCESSING AND AUGMENTATION
def normalize_landmarks(sequence):
    normalized_seq = sequence.copy()
    for frame_idx in range(SEQUENCE_LENGTH):
        #hand 1
        h1_wrist = normalized_seq[frame_idx, 0:3]
        if np.any(h1_wrist):
            normalized_seq[frame_idx, 0:63] -= np.tile(h1_wrist, 21)
            h1_mcp = normalized_seq[frame_idx, 27:30]
            dist = np.linalg.norm(h1_mcp)
            if dist > 0:
                normalized_seq[frame_idx, 0:63] /= dist
            
        #hand 2
        h2_wrist = normalized_seq[frame_idx, 63:66]
        if np.any(h2_wrist):
            normalized_seq[frame_idx, 63:126] -= np.tile(h2_wrist, 21)
            h2_mcp = normalized_seq[frame_idx, 90:93]
            dist = np.linalg.norm(h2_mcp)
            if dist > 0:
                normalized_seq[frame_idx, 63:126] /= dist
            
    return normalized_seq

def rotate_sequence(sequence):
    """Randomly rotates the sequence in 3D space."""
    angle = np.radians(random.uniform(-10, 10))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    
    rotated = sequence.reshape(SEQUENCE_LENGTH, -1, 3)
    rotated = np.dot(rotated, rotation_matrix)
    return rotated.reshape(SEQUENCE_LENGTH, -1)

def augment_sequence(sequence):
    """Aggressive augmentation for generalization."""
    augmented = sequence.copy()
    #Random Rotation
    if random.random() > 0.5:
        augmented = rotate_sequence(augmented)
    #Random Noise
    noise = np.random.normal(0, 0.02, augmented.shape)
    augmented += noise
    #Time-shift
    if random.random() > 0.8:
        augmented = np.roll(augmented, random.randint(-2, 2), axis=0)
    return augmented

class DynamicGestureDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx].numpy()
        if self.augment:
            seq = augment_sequence(seq)
        return torch.FloatTensor(seq), self.labels[idx]

def load_and_preprocess_data():
    sequences, labels = [], []
    label_map = {word: idx for idx, word in enumerate(WORDS)}
    
    for word in WORDS:
        word_dir = os.path.join(DATA_PATH, word)
        if not os.path.exists(word_dir): continue
            
        for file in os.listdir(word_dir):
            if file.endswith(".npy"):
                data = np.load(os.path.join(word_dir, file))
                data = normalize_landmarks(data)
                sequences.append(data)
                labels.append(label_map[word])
                
    return np.array(sequences), np.array(labels), label_map


#MODEL ARCHITECTURE 
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# TRAINING ENGINE
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, label_map = load_and_preprocess_data()
    print(f"Loaded {len(X)} samples. Starting augmented training...")

    # Save mapping
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump({"idx_to_label": {v: k for k, v in label_map.items()}}, f)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = DynamicGestureDataset(X_train, y_train, augment=True)
    test_dataset = DynamicGestureDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seqs), lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, lbls in test_loader:
                seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
                _, pred = torch.max(model(seqs).data, 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()
        
        acc = 100 * correct / total
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {t_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%')
            
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "dynamic_model.pth"))

    print(f"\nTraining Finished. Best Val Accuracy: {best_acc:.2f}%")
    
    # Export optimized ONNX
    print("Exporting optimized ONNX model.")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dynamic_model.pth")))
    model.eval().to("cpu")
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_SIZE)
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            os.path.join(MODEL_DIR, "dynamic_model.onnx"),
            export_params=True,
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'],
            dynamo=False 
        )

if __name__ == "__main__":
    train()
