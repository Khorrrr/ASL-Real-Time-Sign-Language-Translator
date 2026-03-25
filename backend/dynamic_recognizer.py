import os
import json
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger("ASL-Backend")

class DynamicRecognizer:
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dynamic_model")
        model_path = os.path.join(model_dir, "dynamic_model.onnx")
        label_map_path = os.path.join(model_dir, "label_map.json")

        if not os.path.exists(model_path):
            logger.error(f"Dynamic ONNX model not found at {model_path}")
            self.session = None
            return

        with open(label_map_path, "r") as f:
            label_data = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        logger.info("Dynamic Word model loaded (ONNX Runtime)")

    def normalize_landmarks(self, sequence):
        """Translation and Scale Invariant Normalization for live inference."""
        normalized_seq = np.array(sequence).copy()
        for frame_idx in range(30):
            # Hand 1 
            h1_wrist = normalized_seq[frame_idx, 0:3]
            if np.any(h1_wrist):
                normalized_seq[frame_idx, 0:63] -= np.tile(h1_wrist, 21)
                h1_mcp = normalized_seq[frame_idx, 27:30]
                dist = np.linalg.norm(h1_mcp)
                if dist > 0:
                    normalized_seq[frame_idx, 0:63] /= dist
            
            # Hand 2 
            h2_wrist = normalized_seq[frame_idx, 63:66]
            if np.any(h2_wrist):
                normalized_seq[frame_idx, 63:126] -= np.tile(h2_wrist, 21)
                h2_mcp = normalized_seq[frame_idx, 90:93]
                dist = np.linalg.norm(h2_mcp)
                if dist > 0:
                    normalized_seq[frame_idx, 63:126] /= dist
        return normalized_seq

    def predict(self, sequence):
        """
        Predict a word from a sequence of 30 frames.
        """
        if self.session is None:
            return None, 0.0

        normalized_sequence = self.normalize_landmarks(sequence)
        features = normalized_sequence.astype(np.float32).reshape(1, 30, 126)

        ort_inputs = {self.input_name: features}
        ort_outs = self.session.run(None, ort_inputs)
        
        outputs = ort_outs[0][0]
        exp_vals = np.exp(outputs - np.max(outputs))
        probabilities = exp_vals / np.sum(exp_vals)
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]

        word = self.idx_to_label[predicted_idx]

        return word, float(confidence)
