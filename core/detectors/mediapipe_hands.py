# core/detectors/mediapipe_hands.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError(
        "Falta 'mediapipe'. Instala con: pip install mediapipe"
    ) from e

@dataclass
class HandDetection:
    bbox_xyxy: Tuple[int, int, int, int]  # (x1,y1,x2,y2)
    score: float
    handedness: str  # "Left"/"Right"/"Unknown"
    landmarks_xy: List[Tuple[int, int]]  # 21 puntos en pixeles

class HandsDetector:
    """
    Detector de manos con MediaPipe.
    - detect(frame_bgr) -> List[HandDetection]
    """
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        model_complexity: int = 0,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _landmarks_to_bbox(
        self, lmks_norm, w: int, h: int
    ) -> Tuple[Tuple[int, int, int, int], List[Tuple[int, int]]]:
        xs, ys = [], []
        pts: List[Tuple[int, int]] = []
        for l in lmks_norm:
            x = int(l.x * w)
            y = int(l.y * h)
            xs.append(x); ys.append(y)
            pts.append((x, y))
        x1, y1 = max(min(xs), 0), max(min(ys), 0)
        x2, y2 = min(max(xs), w - 1), min(max(ys), h - 1)
        return (x1, y1, x2, y2), pts

    def detect(self, frame_bgr) -> List[HandDetection]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)
        out: List[HandDetection] = []
        if not res.multi_hand_landmarks:
            return out

        handedness_list = []
        if res.multi_handedness:
            for c in res.multi_handedness:
                # "Left"/"Right" con score
                label = c.classification[0].label if c.classification else "Unknown"
                handedness_list.append(label)
        else:
            handedness_list = ["Unknown"] * len(res.multi_hand_landmarks)

        for i, hand_lmks in enumerate(res.multi_hand_landmarks):
            bbox, pts = self._landmarks_to_bbox(hand_lmks.landmark, w, h)
            score = 0.75  # MediaPipe no da score de bbox, usamos un fijo razonable
            handed = handedness_list[i] if i < len(handedness_list) else "Unknown"
            out.append(HandDetection(bbox_xyxy=bbox, score=score, handedness=handed, landmarks_xy=pts))
        return out

def draw_hands(frame_bgr, hands: List[HandDetection]) -> None:
    for det in hands:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(frame_bgr, f"hand {det.handedness}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
