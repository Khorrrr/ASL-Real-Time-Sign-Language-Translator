"""
MediaPipe hand tracking — extracts landmarks from webcam frames.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os


class HandTracker:
    def __init__(self, max_hands=2):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'data', 'processed', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_ts = -1

    def process_frame(self, frame, timestamp_ms):
        """
        Process a frame and return landmarks + hand positions.
        """
        if timestamp_ms <= self.last_ts:
            timestamp_ms = self.last_ts + 1
        self.last_ts = timestamp_ms

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            self.last_ts += 1
            detection_result = self.detector.detect_for_video(mp_image, self.last_ts)

        landmarks = None
        hands_data = [] 
        is_multi_hand = False

        if detection_result.hand_landmarks:
            is_multi_hand = len(detection_result.hand_landmarks) > 1

            for i, hand in enumerate(detection_result.hand_landmarks):
                self._draw_hand(frame, hand)
                
                label = detection_result.handedness[i][0].category_name
                
                thumb = hand[4]
                index = hand[8]
                dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                is_pinched = dist < 0.07

                raw_lms = []
                for lm in hand:
                    raw_lms.append({"x": lm.x, "y": lm.y, "z": lm.z})

                hands_data.append({
                    "label": label,
                    "index_tip": (index.x, index.y),
                    "thumb_tip": (thumb.x, thumb.y),
                    "is_pinched": is_pinched,
                    "pinch_pos": ((thumb.x + index.x)/2, (thumb.y + index.y)/2),
                    "landmarks": raw_lms
                })

            primary_hand = detection_result.hand_landmarks[0]
            raw_landmarks_flat = []
            for lm in primary_hand:
                raw_landmarks_flat.extend([lm.x, lm.y, lm.z])
            landmarks = self._normalize(raw_landmarks_flat)
            
        return landmarks, hands_data, frame, is_multi_hand

    def _draw_hand(self, image, hand_landmarks):
        h, w = image.shape[:2]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
        ]
        for start_idx, end_idx in connections:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            cv2.line(image, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), (0, 255, 0), 2)
        for landmark in hand_landmarks:
            cv2.circle(image, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 0, 255), -1)

    def _normalize(self, landmarks):
        landmarks = np.array(landmarks, dtype=np.float32)
        wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
        normalized = []
        for i in range(0, len(landmarks), 3):
            normalized.append(landmarks[i] - wrist_x)
            normalized.append(landmarks[i + 1] - wrist_y)
            normalized.append(landmarks[i + 2] - wrist_z)
        max_val = max(abs(v) for v in normalized)
        if max_val > 0:
            normalized = [v / max_val for v in normalized]
        return normalized

    def close(self):
        self.detector.close()
