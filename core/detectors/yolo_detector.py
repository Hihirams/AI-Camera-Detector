# core/detectors/yolo_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError(
        "Falta 'ultralytics'. Instala con: pip install ultralytics"
    ) from e

@dataclass
class YoloDet:
    cls: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]

class YOLODetector:
    """
    Wrapper mínimo para YOLOv8 (por defecto personas).
    - detect(frame_bgr) -> List[YoloDet] con cls en {'person', ...}
    """
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = float(conf)
        # Mapa de clases si es el modelo COCO oficial
        self.coco_names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame_bgr) -> List[YoloDet]:
        res = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)
        out: List[YoloDet] = []
        for r in res:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                x1, y1, x2, y2 = map(lambda v: int(max(v, 0)), xyxy)
                score = float(b.conf[0]) if b.conf is not None else 0.0
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                cls_name = self.coco_names.get(cls_id, str(cls_id))
                out.append(YoloDet(cls=cls_name, score=score, bbox_xyxy=(x1, y1, x2, y2)))
        return out

def draw_yolo(frame_bgr, dets: List[YoloDet], only_cls: str | None = None) -> None:
    for d in dets:
        if only_cls and d.cls != only_cls:
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame_bgr, f"{d.cls} {d.score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
