import pytest
import numpy as np
from backend.hand_tracker import HandTracker
from backend.letter_recognizer import LetterRecognizer
import os

def test_hand_tracker_init():
    """Test that HandTracker initializes without errors."""
    tracker = HandTracker(max_hands=1)
    assert tracker is not None

def test_letter_recognizer_init():
    """Test that LetterRecognizer initializes."""
    model_path = "models/letter_model/asl_model.onnx"
    label_path = "models/letter_model/label_map.json"
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        recognizer = LetterRecognizer(model_path=model_path, label_map_path=label_path)
        assert recognizer is not None
    else:
        pytest.skip("Model files not found for testing.")

def test_preprocess_landmarks():
    """Test that landmark preprocessing returns the correct shape."""
    mock_landmarks = [{"x": i, "y": i, "z": i} for i in range(21)]
    
    model_path = "models/letter_model/asl_model.onnx"
    label_path = "models/letter_model/label_map.json"
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        recognizer = LetterRecognizer(model_path=model_path, label_map_path=label_path)
        processed = recognizer.preprocess(mock_landmarks)
        assert processed.shape == (1, 63)
    else:
        pytest.skip("Model files not found for testing.")
