# apps/vision_loop.py
from __future__ import annotations
import argparse, time, cv2, numpy as np
from typing import List, Tuple
from core.io_utils import repo_path, load_yaml, load_roi_polygons  # ROI + paths
from core.video_source import VideoSource, SourceConfig            # Fuente de video
from core.event_store import EventStore, FrameRing, write_clip_from_buffer  # Clips/Eventos

# Reglas iniciales
RULE_HAND_IN_ROI = "HAND_IN_ROI"
SEVERITY = "HIGH"

def load_camera_cfg(camera_id: str) -> SourceConfig:
    cfg_path = repo_path("configs", "cameras.yaml")
    cfg = load_yaml(cfg_path)
    cameras = cfg.get("cameras", {})
    if camera_id not in cameras:
        raise RuntimeError(f"camera_id '{camera_id}' no encontrado en {cfg_path}")
    c = cameras[camera_id]
    return SourceConfig(
        url=c.get("source"),
        width=c.get("width"),
        height=c.get("height"),
        fps_target=c.get("fps_target"),
    )

def draw_roi(frame, pts: np.ndarray | None):
    disp = frame.copy()
    if pts is not None and len(pts) >= 3:
        cv2.polylines(disp, [pts], isClosed=True, color=(0,0,255), thickness=2)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [pts], (0,0,255))
        disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)
    return disp

def inside_roi(pt_xy: Tuple[int,int], roi_pts: np.ndarray | None) -> bool:
    if roi_pts is None or len(roi_pts) < 3:
        return True  # si no hay ROI, considerar todo dentro (útil para pruebas)
    x, y = pt_xy
    return cv2.pointPolygonTest(roi_pts, (float(x), float(y)), False) >= 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera_id en configs/cameras.yaml (ej: CAM_DEMO)")
    ap.add_argument("--model", default="hand", choices=["hand","yolo"], help="Detector a usar")
    ap.add_argument("--yolo_path", default="yolov8n.pt", help="Ruta del modelo YOLO (si --model yolo)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confianza mínima")
    ap.add_argument("--pre_s", type=float, default=6.0, help="segundos de pre-evento")
    ap.add_argument("--post_s", type=float, default=6.0, help="segundos post-evento")
    args = ap.parse_args()

    # Cargar fuente
    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg)
    if not src.open():
        raise RuntimeError("No se pudo abrir la fuente de video")

    # Primer frame
    ok, frame = src.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer frame inicial")
    h, w = frame.shape[:2]

    # Cargar ROI
    polygons = load_roi_polygons(args.camera)
    roi_pts = np.array(polygons[0], dtype=np.int32) if polygons else None

    # Detector
    detector = None
    draw_fn = None
    if args.model == "hand":
        from core.detectors.mediapipe_hands import HandsDetector, draw_hands
        detector = HandsDetector(min_detection_confidence=args.conf)
        draw_fn = draw_hands
    else:
        from core.detectors.yolo_detector import YOLODetector, draw_yolo
        detector = YOLODetector(model_path=args.yolo_path, conf=args.conf)
        draw_fn = lambda im, dets: draw_yolo(im, dets, only_cls=None)

    # Buffers/eventos
    fps_assumed = int(scfg.fps_target or 25)
    ring = FrameRing(max(1, int(args.pre_s * fps_assumed)))
    post_needed = max(1, int(args.post_s * fps_assumed))
    cooldown = 0
    store = EventStore(repo_path())

    cv2.namedWindow("AI Camera - Vision", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            break
        ts = time.time()
        ring.push(ts, frame)

        disp = draw_roi(frame, roi_pts)

        # ---- Detección ----
        detections = detector.detect(frame)

        # Normalizamos lista de puntos centrales para checar ROI
        centers: List[Tuple[int,int]] = []
        if args.model == "hand":
            # MediaPipe: lista de HandDetection
            for d in detections:
                x1, y1, x2, y2 = d.bbox_xyxy
                centers.append(((x1 + x2)//2, (y1 + y2)//2))
        else:
            # YOLO: lista de YoloDet; por ahora buscamos 'person' o TODO: hand-model si se agrega
            for d in detections:
                if d.cls == "person":
                    x1, y1, x2, y2 = d.bbox_xyxy
                    centers.append(((x1 + x2)//2, (y1 + y2)//2))

        # dibujar detecciones
        try:
            draw_fn(disp, detections)
        except Exception:
            pass  # fallback silencioso si el draw cambia de firma

        # Regla: HAND_IN_ROI (o persona como aproximación si model=yolo)
        trigger = any(inside_roi(c, roi_pts) for c in centers)

        if trigger and cooldown == 0:
            # Capturamos post-evento
            post_frames: List[np.ndarray] = []
            for _ in range(post_needed):
                ok2, fr2 = src.read()
                if not ok2 or fr2 is None:
                    break
                post_frames.append(fr2)
                show = draw_roi(fr2, roi_pts)
                cv2.putText(show, f"ALERTA: {RULE_HAND_IN_ROI}", (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                cv2.imshow("AI Camera - Vision", show)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    src.release(); cv2.destroyAllWindows(); return

            # Escribir clip
            out_path = store.clip_path(args.camera, RULE_HAND_IN_ROI)
            write_clip_from_buffer(out_path, ring.dump(), post_frames, fps_assumed)

            # Guardar evento
            meta = {
                "model": args.model,
                "detections": len(detections),
                "roi": "roi_1",
            }
            store.save_event(args.camera, RULE_HAND_IN_ROI, SEVERITY, str(out_path), meta)

            cooldown = fps_assumed * 2  # 2s
            cv2.putText(disp, f"ALERTA: {RULE_HAND_IN_ROI}", (16, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        if cooldown > 0:
            cooldown -= 1
            cv2.putText(disp, f"cooldown: {cooldown/fps_assumed:.1f}s",
                        (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("AI Camera - Vision", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
