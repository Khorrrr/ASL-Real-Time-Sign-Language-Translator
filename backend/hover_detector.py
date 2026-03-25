"""
Detects when the user's hand hovers over UI elements.
"""


class HoverDetector:
    def __init__(self):
        self.suggestion_boxes = []
        self.hovered_word = None
        self.hover_frames = 0
        self.HOVER_CONFIRM_FRAMES = 5 
        self.cooldown_frames = 0

    def update_suggestion_boxes(self, words, num_boxes=5):
        self.suggestion_boxes = []
        if not words: return

        box_width = 0.14
        gap = 0.015
        total_width = len(words) * box_width + (len(words) - 1) * gap
        start_x = max(0.05, (1.0 - total_width) / 2)

        for i, word in enumerate(words):
            x_min = start_x + i * (box_width + gap)
            x_max = x_min + box_width
            if x_max > 0.98: x_max = 0.98
            
            self.suggestion_boxes.append({
                "word": word,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": 0.78,
                "y_max": 0.96,
                "index": i
            })

    def check_hover(self, index_tips, is_multi_hand=False):
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return None

        if not index_tips:
            self.hovered_word = None
            self.hover_frames = 0
            return None

        HOVER_THRESHOLD = 3 if is_multi_hand else 6

        for tip_x, tip_y in index_tips:
            if tip_x is None or tip_y is None:
                continue

            for box in self.suggestion_boxes:
                if (box["x_min"] <= tip_x <= box["x_max"] and
                        box["y_min"] <= tip_y <= box["y_max"]):

                    if self.hovered_word == box["word"]:
                        self.hover_frames += 1
                    else:
                        self.hovered_word = box["word"]
                        self.hover_frames = 1

                    if self.hover_frames >= HOVER_THRESHOLD:
                        self.cooldown_frames = 15 
                        return {
                            "type": "accepted",
                            "word": box["word"],
                            "index": box["index"],
                            "progress": 1.0
                        }

                    return {
                        "type": "suggestion",
                        "word": box["word"],
                        "index": box["index"],
                        "progress": min(self.hover_frames / HOVER_THRESHOLD, 1.0)
                    }

        self.hovered_word = None
        self.hover_frames = 0
        return None
