"""
Letter stabilization — prevents flickering predictions.
Only accepts a letter after seeing it consistently for N frames.
"""


class LetterStabilizer:
    def __init__(self, required_frames=8, cooldown_frames=5):
        self.required_frames = required_frames
        self.cooldown_frames = cooldown_frames

        self.current_prediction = None
        self.prediction_count = 0
        self.last_accepted = None
        self.cooldown_counter = 0

    def update(self, prediction, confidence, min_confidence=0.7):
        """Feed a new prediction. Returns accepted letter or None."""
        if confidence < min_confidence:
            self.prediction_count = 0
            self.current_prediction = None
            return None

        if prediction == "nothing":
            self.prediction_count = 0
            self.current_prediction = None
            return None

            self.cooldown_counter -= 1
            return None

            self.prediction_count += 1
        else:
            self.current_prediction = prediction
            self.prediction_count = 1

            self.last_accepted = prediction
            self.prediction_count = 0
            self.cooldown_counter = self.cooldown_frames
            return prediction

        return None

    def reset(self):
        self.current_prediction = None
        self.prediction_count = 0
        self.last_accepted = None
        self.cooldown_counter = 0