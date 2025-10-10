# core/event_store.py
from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import os

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

class EventStore:
    """
    Guarda eventos en CSV y clips en /data/clips.
    """
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.events_csv = ensure_dir(repo_root / "data").joinpath("events.csv")
        self.clips_dir = ensure_dir(repo_root / "data" / "clips")

        # crear encabezado si no existe
        if not self.events_csv.exists():
            with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts_utc", "camera_id", "rule_id", "severity", "clip_path", "meta"])

    def save_event(self, camera_id: str, rule_id: str, severity: str, clip_path: str, meta: Dict[str, Any]):
        ts = int(time.time())
        with open(self.events_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, camera_id, rule_id, severity, clip_path, meta])

    def clip_path(self, camera_id: str, rule_id: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        name = f"{camera_id}_{rule_id}_{ts}.mp4"
        return self.clips_dir / name

class FrameRing:
    """
    Buffer circular de frames (para guardar pre-evento).
    Guarda (ts, frame_bgr).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: List[Tuple[float, Any]] = []
        self.idx = 0

    def push(self, ts: float, frame):
        if len(self.buf) < self.capacity:
            self.buf.append((ts, frame.copy()))
        else:
            self.buf[self.idx] = (ts, frame.copy())
            self.idx = (self.idx + 1) % self.capacity

    def dump(self) -> List[Tuple[float, Any]]:
        # orden cronológico
        if len(self.buf) < self.capacity:
            return list(self.buf)
        return self.buf[self.idx:] + self.buf[:self.idx]

def write_clip_from_buffer(out_path: Path, buffer_frames: List[Tuple[float, Any]], post_frames: List[Any], fps: int):
    if not buffer_frames and not post_frames:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # inferir tamaño a partir del primer frame disponible
    sample = (buffer_frames[0][1] if buffer_frames else post_frames[0])
    h, w = sample.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _, fr in buffer_frames:
        vw.write(fr)
    for fr in post_frames:
        vw.write(fr)
    vw.release()
