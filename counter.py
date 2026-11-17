import math
from collections import Counter, deque

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
# (pip, tip) for index, middle, ring, pinky
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]

THUMB_SPREAD = 0.6  # thumb tip to index base, as a fraction of palm length


class HandDetector:
    def __init__(self, model_path, max_hands=1, min_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, rgb, timestamp_ms):
        """Returns a list of hands, each a list of 21 (x, y) points in pixels."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        h, w = rgb.shape[:2]
        return [[(p.x * w, p.y * h) for p in hand] for hand in result.hand_landmarks]

    def close(self):
        self.landmarker.close()


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extended_fingers(pts):
    """Returns 5 booleans (thumb..pinky). Distance-based, so it works for any hand rotation."""
    wrist = pts[WRIST]
    states = [
        _dist(pts[THUMB_TIP], pts[PINKY_MCP]) > _dist(pts[THUMB_IP], pts[PINKY_MCP])
        and _dist(pts[THUMB_TIP], pts[INDEX_MCP]) > THUMB_SPREAD * _dist(wrist, pts[MIDDLE_MCP])
    ]
    states += [_dist(wrist, pts[tip]) > _dist(wrist, pts[pip]) for pip, tip in FINGERS]
    return states


class Smoother:
    def __init__(self, size=5):
        self.history = deque(maxlen=size)

    def push(self, value):
        self.history.append(value)
        return Counter(self.history).most_common(1)[0][0]
