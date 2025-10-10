from dataclasses import dataclass
from typing import Optional, Tuple, Any
import time, cv2

@dataclass
class SourceConfig:
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps_target: Optional[float] = None

class VideoSource:
    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg; self.cap = None; self._last_ts = 0.0

    def open(self) -> bool:
        url = self.cfg.url
        if url.startswith("file://"): src = url.replace("file://","",1)
        elif url.startswith("webcam://"): src = int(url.replace("webcam://","",1))
        else: src = url
        self.cap = cv2.VideoCapture(src)
        if not self.cap or not self.cap.isOpened():
            print(f"[VideoSource] No se pudo abrir: {url}"); return False
        if self.cfg.width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        if self.cfg.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        return True

    def read(self) -> Optional[Tuple[bool, Any]]:
        if not self.cap: return None
        ok, frame = self.cap.read()
        if not ok: return (False, None)
        if self.cfg.fps_target and self.cfg.fps_target > 0:
            now = time.time(); min_dt = 1.0/float(self.cfg.fps_target)
            elapsed = now - self._last_ts
            if elapsed < min_dt: time.sleep(max(0.0, min_dt - elapsed))
            self._last_ts = time.time()
        return (True, frame)

    def release(self):
        if self.cap: self.cap.release(); self.cap = None
