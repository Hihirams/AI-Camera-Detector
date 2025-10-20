"""
apps/multi_vision.py  (v2 – LIVE detections + ROI alerts)

- Abre cámaras desde configs/cameras.yaml (file:// mp4 en loop, o RTSP/HTTP/USB).
- Corre YOLOv8 por cámara (clase 'person' = 0) y dibuja cajas/labels en el mosaico.
- Carga ROI por cámara desde configs/rois/<CAM_ID>.json (polígonos).
- Si una persona intersecta/cae dentro del ROI => borde rojo + texto ALERT en la celda.
- Layout 2x2 / 3x3 con --cols, tamaño de celdas configurable.

Uso:
  python apps/multi_vision.py --cols 3 --yolo_path yolov8n.pt --conf 0.35

Teclas:
  Q / ESC  -> salir
"""

from __future__ import annotations
import os, sys, time, json, math, threading, queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

# ---------- Rutas base ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
CAMERAS_YAML = CONFIGS / "cameras.yaml"
ROIS_DIR = CONFIGS / "rois"

# ---------- Utils ----------
def resolve_source(url: str) -> str:
    if url.startswith("file://"):
        # Eliminar prefijo file://
        path_str = url.replace("file:///", "").replace("file://", "")
        path = Path(path_str)
        
        # Si es relativa, hacerla absoluta respecto a REPO_ROOT
        if not path.is_absolute():
            path = REPO_ROOT / path
        
        return str(path)
    return url

def load_yaml_cameras(path: Path) -> Dict[str, Dict[str, Any]]:
    cams: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cams
    current_cam = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if line.endswith(":") and not line.startswith(" "):
                continue
            if line.endswith(":") and line.startswith("  "):
                current_cam = line.strip()[:-1]
                cams.setdefault(current_cam, {})
            elif ":" in line and line.startswith("    "):
                k, v = [x.strip() for x in line.strip().split(":", 1)]
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                else:
                    try:
                        if "." in v:
                            v = float(v)
                            if v.is_integer():
                                v = int(v)
                        else:
                            v = int(v)
                    except Exception:
                        pass
                if current_cam:
                    cams[current_cam][k] = v
    return cams

def draw_border(img, color=(0,255,0), thickness=3):
    cv2.rectangle(img, (0,0), (img.shape[1]-1, img.shape[0]-1), color, thickness)

def poly_point_inside(poly: np.ndarray, pt: Tuple[float,float]) -> bool:
    # Ray casting
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-9) + x1):
            inside = not inside
    return inside

def rect_poly_intersect(rect: Tuple[int,int,int,int], poly: np.ndarray) -> bool:
    # rect = (x1,y1,x2,y2)
    x1,y1,x2,y2 = rect
    # quick reject
    if x2 < poly[:,0].min() or x1 > poly[:,0].max() or y2 < poly[:,1].min() or y1 > poly[:,1].max():
        # maybe still intersects if poly fully inside rect
        pass
    # test: center point inside poly OR any corner inside poly OR bbox overlaps polygon bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    if poly_point_inside(poly, (cx,cy)): return True
    for px,py in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]:
        if poly_point_inside(poly, (px,py)): return True
    # coarse bbox overlap
    Rx1,Ry1,Rx2,Ry2 = x1,y1,x2,y2
    Px1,Py1,Px2,Py2 = int(poly[:,0].min()), int(poly[:,1].min()), int(poly[:,0].max()), int(poly[:,1].max())
    if Rx1 <= Px2 and Rx2 >= Px1 and Ry1 <= Py2 and Ry2 >= Py1:
        return True
    return False

def load_roi_polygon(cam_id: str, target_w: int, target_h: int) -> Optional[np.ndarray]:
    """
    Espera configs/rois/CAM_ID.json con:
      {"polygons":[[[x,y],...]], "normalized": true|false}
    Si normalized=true, x,y en 0..1 y se escalan a target_w/h.
    Toma solo el primer polígono para 'zona prohibida' (puedes extender a varios).
    """
    p = ROIS_DIR / f"{cam_id}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        polys = obj.get("polygons") or obj.get("roi") or []
        if not polys: return None
        poly = polys[0]
        norm = bool(obj.get("normalized", True))
        pts = []
        for x,y in poly:
            if norm:
                pts.append((int(x*target_w), int(y*target_h)))
            else:
                pts.append((int(x), int(y)))
        return np.array(pts, dtype=np.int32)
    except Exception:
        return None

# ---------- Detector YOLO ----------
class PersonDetector:
    def __init__(self, yolo_path: str, conf: float = 0.35, device: str = ""):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Falta 'ultralytics'. Instala con: pip install ultralytics") from e
        self.model = YOLO(yolo_path)
        self.conf = float(conf)
        self.device = device  # '', 'cpu', 'cuda'
    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Devuelve lista de bboxes de personas: (x1,y1,x2,y2,score)
        """
        res = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            verbose=False,
            max_det=50,
            device=self.device if self.device else None
        )
        out = []
        for r in res:
            if not hasattr(r, "boxes") or r.boxes is None: 
                continue
            for b in r.boxes:
                cls = int(b.cls.item()) if hasattr(b, "cls") else -1
                if cls != 0:  # 'person' = 0 en COCO
                    continue
                xyxy = b.xyxy[0].tolist()
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                out.append((x1,y1,x2,y2,conf))
        return out

# ---------- Worker por cámara ----------
class CamWorker(threading.Thread):
    def __init__(self, cam_id: str, source: str, out_queue: "queue.Queue[Tuple[str, Any]]",
                 cell_size: Tuple[int,int], detector: Optional[PersonDetector], roi_poly: Optional[np.ndarray]):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source_url = resolve_source(source)
        self.out_queue = out_queue
        self.w, self.h = cell_size
        self.detector = detector
        self.roi_poly = roi_poly
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.source_url)
            if not cap or not cap.isOpened():
                self.out_queue.put((self.cam_id, self._banner("no signal")))
                time.sleep(1.0)
                continue

            is_file = os.path.exists(self.source_url) or self.source_url.lower().endswith((".mp4",".avi",".mov",".mkv",".m4v"))

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:
                        cap.release()
                        cap = cv2.VideoCapture(self.source_url)
                        continue
                    else:
                        break

                # resize a nuestra celda
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

                alert_active = False
                # detección inline (si hay detector)
                if self.detector is not None:
                    boxes = self.detector.detect(frame)
                    for (x1,y1,x2,y2,score) in boxes:
                        color = (60, 220, 60)
                        if self.roi_poly is not None and rect_poly_intersect((x1,y1,x2,y2), self.roi_poly):
                            color = (0, 0, 255)
                            alert_active = True
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, f"person {score:.2f}", (x1, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # dibuja ROI si existe
                    if self.roi_poly is not None:
                        cv2.polylines(frame, [self.roi_poly], True, (0,255,255), 2)

                self.out_queue.put((self.cam_id, frame, alert_active))

                # pequeño yield
                if cv2.waitKey(1) & 0xFF == 27:
                    self._stop.set(); break

            cap.release()

    def _banner(self, text: str):
        img = np.zeros((self.h, self.w, 3), dtype="uint8")
        cv2.putText(img, f"{self.cam_id}: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return img

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-Vision (LIVE) – YOLO + ROI alerts")
    ap.add_argument("--cols", type=int, default=3, help="Columnas del mosaico (2-4 recomendado)")
    ap.add_argument("--cell_w", type=int, default=480, help="Ancho de celda")
    ap.add_argument("--cell_h", type=int, default=270, help="Alto de celda")

    # Detección
    ap.add_argument("--enable_det", type=int, default=1, help="1=activar detección YOLO inline, 0=solo video")
    ap.add_argument("--yolo_path", type=str, default=str(REPO_ROOT / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", type=str, default="", help="''/cpu/cuda (si disponible)")

    args = ap.parse_args()

    cams = load_yaml_cameras(CAMERAS_YAML)
    if not cams:
        print(f"[ERR] No cameras found in {CAMERAS_YAML}")
        return

    cam_items = sorted(cams.items(), key=lambda kv: kv[0])
    n = len(cam_items)
    rows = int(math.ceil(n / max(1,args.cols)))
    cols = max(1, args.cols)
    W, H = cols*args.cell_w, rows*args.cell_h

    # detector por cámara (cargar uno por worker para evitar lock de threads)
    detector_template = None
    if args.enable_det:
        try:
            detector_template = PersonDetector(args.yolo_path, conf=args.conf, device=args.device)
        except Exception as e:
            print("[WARN] YOLO no disponible:", e)
            detector_template = None

    # ROI por cámara
    roi_by_cam: Dict[str, Optional[np.ndarray]] = {}
    for cam_id, _ in cam_items:
        roi_by_cam[cam_id] = load_roi_polygon(cam_id, args.cell_w, args.cell_h)

    q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=64)
    workers: List[CamWorker] = []
    last_frames: Dict[str, Tuple[np.ndarray, bool]] = {}  # cam -> (frame, alert)

    for cam_id, cfg in cam_items:
        src = cfg.get("source") or ""
        det_inst = None
        if detector_template is not None:
            # crea un detector "ligero" por hilo (comparten pesos internamente; si no, igual funciona CPU)
            det_inst = PersonDetector(args.yolo_path, conf=args.conf, device=args.device)
        worker = CamWorker(cam_id, resolve_source(src), q, (args.cell_w, args.cell_h), det_inst, roi_by_cam[cam_id])
        worker.start()
        workers.append(worker)

    win = "Multi-Vision (LIVE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    try:
        while True:
            # drenar queue
            try:
                for _ in range(8):
                    item = q.get_nowait()
                    if len(item) == 3:
                        cam_id, frame, alert = item
                    else:
                        cam_id, frame = item
                        alert = False
                    last_frames[cam_id] = (frame, alert)
            except queue.Empty:
                pass

            canvas = np.zeros((H, W, 3), dtype="uint8")
            for idx, (cam_id, cfg) in enumerate(cam_items):
                r0 = (idx // cols) * args.cell_h
                c0 = (idx % cols) * args.cell_w
                frame, alert = last_frames.get(cam_id, (None, False))
                cell = np.zeros((args.cell_h, args.cell_w, 3), dtype="uint8")
                if frame is not None:
                    cell[:] = frame
                # título + borde
                cv2.putText(cell, cam_id, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                draw_border(cell, (0,0,255) if alert else (60,120,60), 3)
                if alert:
                    cv2.putText(cell, "ALERT: PERSON IN ROI", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                canvas[r0:r0+args.cell_h, c0:c0+args.cell_w] = cell

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break

    finally:
        for w in workers:
            w.stop()
        time.sleep(0.2)
        cv2.destroyWindow(win)

if __name__ == "__main__":
    import math
    main()
