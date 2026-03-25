"""
Loads the trained ASL model and predicts letters from landmarks using ONNX Runtime.
"""

import os
import json
import pickle
import numpy as np
import sys
import onnxruntime as ort
import logging

logger = logging.getLogger("ASL-Backend")

class LetterRecognizer:
    def __init__(self, model_path=None, label_map_path=None):
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letter_model")

        if not model_path:
            model_path = os.path.join(model_dir, "asl_model.onnx")
        if not label_map_path:
            label_map_path = os.path.join(model_dir, "label_map.json")

        if not os.path.exists(model_path):
            logger.warning(f"ONNX model not found at {model_path}. Trying fallback path...")
            model_path = os.path.join(model_dir, "asl_model.onnx")

        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        with open(label_map_path, "r") as f:
            label_data = json.load(f)

        self.idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}

        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        logger.info("Letter model loaded (ONNX Runtime)")

    def preprocess(self, landmarks):
        """Preprocesses dictionary-based landmarks into flat array."""
        if isinstance(landmarks[0], dict):
             flat_landmarks = []
             for lm in landmarks:
                 flat_landmarks.extend([lm["x"], lm["y"], lm["z"]])
             return np.array(flat_landmarks).reshape(1, -1)
        return np.array(landmarks).reshape(1, -1)

    def predict(self, landmarks):
        """Predict a letter from 63 landmark values. Returns: (letter, confidence)"""
        if isinstance(landmarks[0], dict):
             features = self.preprocess(landmarks)
        else:
             features = np.array(landmarks).reshape(1, -1)

        features = self.scaler.transform(features).astype(np.float32)

        ort_inputs = {self.input_name: features}
        ort_outs = self.session.run(None, ort_inputs)

        outputs = ort_outs[0][0]
        exp_vals = np.exp(outputs - np.max(outputs))
        probabilities = exp_vals / np.sum(exp_vals)

        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]

        letter = self.idx_to_label[predicted_idx]

        return letter, float(confidence)