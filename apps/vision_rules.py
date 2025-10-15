# apps/vision_rules.py
from __future__ import annotations
from pathlib import Path
import time, cv2, os
import argparse, cv2, numpy as np
from typing import Dict, List, Tuple
from core.io_utils import repo_path, load_yaml, load_roi_polygons
from core.video_source import VideoSource, SourceConfig
from core.event_store import EventStore, FrameRing, write_clip_from_buffer
from core.rules.engine import load_rules, required_models, get_roi_points_for_camera
from core.detectors.yolo_detector import draw_yolo
from core.detectors.mediapipe_hands import draw_hands

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "outputs" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiter por cámara para escribir frames ~5 fps
_LIVE_DUMP_LAST: dict[str, float] = {}

def write_live_overlay(camera_id: str, annotated_frame, fps_limit: float = 5.0):
    """
    Guarda un JPG por cámara en outputs/live/CAM_ID.jpg para que multi_vision lo muestre.
    Limita a ~fps_limit (default 5 fps) para no saturar el disco.
    """
    if annotated_frame is None:
        return
    # seguridad ante valores inválidos
    fps_limit = max(float(fps_limit or 5.0), 0.5)
    period = 1.0 / fps_limit

    now_ts = time.time()
    last_ts = _LIVE_DUMP_LAST.get(camera_id, 0.0)
    if (now_ts - last_ts) < period:
        return
    try:
        out_path = LIVE_DIR / f"{camera_id}.jpg"
        cv2.imwrite(str(out_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _LIVE_DUMP_LAST[camera_id] = now_ts
    except Exception:
        # no rompemos el loop de inferencia por errores de disco
        pass

def load_camera_cfg(camera_id: str) -> SourceConfig:
    cfg_path = repo_path("configs", "cameras.yaml")
    cfg = load_yaml(cfg_path)
    cams = cfg.get("cameras", {})
    if camera_id not in cams:
        raise RuntimeError(f"camera_id '{camera_id}' no encontrado en {cfg_path}")
    c = cams[camera_id]
    return SourceConfig(
        url=c.get("source"),
        width=c.get("width"),
        height=c.get("height"),
        fps_target=c.get("fps_target"),
    )

def draw_roi(frame, pts: np.ndarray | None):
    disp = frame.copy()
    if pts is not None and len(pts) >= 3:
        cv2.polylines(disp, [pts], True, (0,0,255), 2)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [pts], (0,0,255))
        disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)
    return disp

def inside_roi(pt_xy: Tuple[int,int], roi_pts: np.ndarray | None) -> bool:
    if roi_pts is None or len(roi_pts) < 3:
        return True
    x, y = pt_xy
    return cv2.pointPolygonTest(roi_pts, (float(x), float(y)), False) >= 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="ID en configs/cameras.yaml (ej: CAM_DEMO)")
    ap.add_argument("--conf", type=float, default=0.35, help="conf por defecto si la regla no trae override")
    ap.add_argument("--yolo_path", default="yolov8n.pt", help="ruta del modelo YOLO si alguna regla usa yolo")
    ap.add_argument("--pre_s", type=float, default=6.0)
    ap.add_argument("--post_s", type=float, default=6.0)
    args = ap.parse_args()

    # Reglas
    rules = load_rules()
    if not rules:
        raise RuntimeError("No hay reglas en configs/rules.yaml")

    # Cámara
    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg)
    if not src.open():
        raise RuntimeError("No se pudo abrir la fuente de video")
    ok, frame = src.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer frame inicial")
    h, w = frame.shape[:2]

    # ROI (por ahora: un ROI por cámara; si agregas varios, ampliamos)
    roi_pts_list = get_roi_points_for_camera(args.camera, "roi_1")
    roi_pts = np.array(roi_pts_list, dtype=np.int32) if roi_pts_list else None

    # Detectores requeridos por las reglas
    models = required_models(rules)
    detectors: Dict[str, object] = {}
    if "hand" in models:
        from core.detectors.mediapipe_hands import HandsDetector
        conf_hand = max((r.conf_override or args.conf) for r in rules if r.model == "hand")
        detectors["hand"] = HandsDetector(min_detection_confidence=conf_hand)
    if "yolo" in models:
        from core.detectors.yolo_detector import YOLODetector
        conf_yolo = max((r.conf_override or args.conf) for r in rules if r.model == "yolo")
        detectors["yolo"] = YOLODetector(model_path=args.yolo_path, conf=conf_yolo)

    # Buffers y store
    fps_assumed = int(scfg.fps_target or 25)
    ring = FrameRing(max(1, int(args.pre_s * fps_assumed)))
    post_needed = max(1, int(args.post_s * fps_assumed))
    cooldown = 0
    store = EventStore(repo_path())

    cv2.namedWindow("AI Camera - Rules", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            break
        ts = time.time()
        ring.push(ts, frame)

        disp = draw_roi(frame, roi_pts)

        # Ejecutar detecciones por modelo una sola vez por frame
        detections: Dict[str, list] = {}
        for m, det in detectors.items():
            detections[m] = det.detect(frame)

        # Dibujar detecciones en disp para el overlay
        if "hand" in detections:
            draw_hands(disp, detections["hand"])
        if "yolo" in detections:
            draw_yolo(disp, detections["yolo"], only_cls="person")

        # Construir centros por tipo estandarizado
        centers_by_model: Dict[str, List[Tuple[int,int]]] = {m: [] for m in models}
        # Hands → bbox centro
        if "hand" in detections:
            for d in detections["hand"]:
                x1, y1, x2, y2 = d.bbox_xyxy
                centers_by_model["hand"].append(((x1 + x2)//2, (y1 + y2)//2))
        # YOLO → solo clase "person" por ahora
        if "yolo" in detections:
            for d in detections["yolo"]:
                if getattr(d, "cls", None) == "person" or getattr(d, "name", None) == "person":
                    x1, y1, x2, y2 = d.bbox_xyxy
                    centers_by_model["yolo"].append(((x1 + x2)//2, (y1 + y2)//2))

        # Evaluar reglas
        triggered: List[str] = []
        for r in rules:
            cms = centers_by_model.get(r.model, [])
            if any(inside_roi(c, roi_pts) for c in cms):
                triggered.append(r.id)

        # Si alguna regla dispara y no estamos en cooldown → capturamos clip + evento(s)
        if triggered and cooldown == 0:
            post_frames: List[np.ndarray] = []
            for _ in range(post_needed):
                ok2, fr2 = src.read()
                if not ok2 or fr2 is None:
                    break
                post_frames.append(fr2)
                show = draw_roi(fr2, roi_pts)
                cv2.putText(show, f"ALERTA: {', '.join(triggered)}", (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                cv2.imshow("AI Camera - Rules", show)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    src.release(); cv2.destroyAllWindows(); return

            # Escribir clip una vez y registrar múltiples reglas si quieres
            out_path = store.clip_path(args.camera, triggered[0])
            write_clip_from_buffer(out_path, ring.dump(), post_frames, fps_assumed)

            for rid in triggered:
                store.save_event(
                    args.camera,
                    rid,
                    "HIGH",  # puedes mapear desde r.severity si lo necesitas
                    str(out_path),
                    {"models": list(models), "roi": "roi_1", "rules": triggered},
                )

            cooldown = fps_assumed * 2
            cv2.putText(disp, f"ALERTA: {', '.join(triggered)}", (16, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        if cooldown > 0:
            cooldown -= 1
            cv2.putText(disp, f"cooldown: {cooldown/fps_assumed:.1f}s",
                        (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("AI Camera - Rules", disp)

        write_live_overlay(args.camera, disp, fps_limit=5.0)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
