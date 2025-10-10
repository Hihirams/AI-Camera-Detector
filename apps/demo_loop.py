# apps/demo_loop.py
from __future__ import annotations
import argparse, time, cv2, numpy as np
from pathlib import Path
from core.io_utils import repo_path, load_yaml, load_roi_polygons
from core.video_source import VideoSource, SourceConfig
from core.event_store import EventStore, FrameRing, write_clip_from_buffer

RULE_ID = "MOTION_IN_ROI"
SEVERITY = "MEDIUM"

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

def draw_roi(frame, pts):
    disp = frame.copy()
    if pts is not None and len(pts) >= 3:
        cv2.polylines(disp, [pts], isClosed=True, color=(0,0,255), thickness=2)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [pts], (0,0,255))
        disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)
    return disp

def mask_roi(shape_hw, pts):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if pts is not None and len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera_id (cameras.yaml)")
    ap.add_argument("--pre_s", type=float, default=6.0, help="segundos de pre-evento")
    ap.add_argument("--post_s", type=float, default=6.0, help="segundos post-evento")
    ap.add_argument("--motion_area_px", type=int, default=1500, help="área mínima de movimiento para alertar")
    args = ap.parse_args()

    # cargar fuente
    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg)
    if not src.open():
        raise RuntimeError("No se pudo abrir la fuente")

    # leer primer frame para preparar estructuras
    ok, frame = src.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer frame inicial")
    h, w = frame.shape[:2]

    # cargar ROI
    polygons = load_roi_polygons(args.camera)
    if not polygons:
        print("[WARN] No hay ROI para esta cámara. Dibuja una con roi_editor.")
        roi_pts = None
    else:
        roi_pts = np.array(polygons[0], dtype=np.int32)

    roi_mask = mask_roi((h, w), roi_pts) if roi_pts is not None else np.ones((h, w), dtype=np.uint8)*255

    # bg-subtractor para movimiento
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

    # buffer circular de pre-evento
    fps_assumed = int(scfg.fps_target or 25)
    pre_capacity = max(1, int(args.pre_s * fps_assumed))
    ring = FrameRing(pre_capacity)

    # para post-evento
    post_needed = max(1, int(args.post_s * fps_assumed))
    post_frames: list = []
    cooldown = 0  # frames de espera para no spamear

    # almacenamiento de eventos y clips
    store = EventStore(repo_path())

    cv2.namedWindow("AI Camera Demo", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            break

        ts = time.time()
        ring.push(ts, frame)

        # detección de movimiento (solo dentro de ROI)
        fg = bg.apply(frame)
        fg = cv2.bitwise_and(fg, fg, mask=roi_mask)
        fg = cv2.medianBlur(fg, 5)
        _, th = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in cnts)

        disp = draw_roi(frame, roi_pts)
        cv2.putText(disp, f"motion_area: {int(motion_area)} px", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        alert = motion_area >= args.motion_area_px and cooldown == 0

        if alert:
            # congelar un post-buffer de N frames
            post_frames = []
            for _ in range(post_needed):
                ok2, fr2 = src.read()
                if not ok2 or fr2 is None:
                    break
                post_frames.append(fr2)
                # también mostrar mientras rellenamos post
                show = draw_roi(fr2, roi_pts)
                cv2.putText(show, "ALERTA: Movimiento en ROI", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                cv2.imshow("AI Camera Demo", show)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    src.release(); cv2.destroyAllWindows(); return

            # escribir clip
            out_path = store.clip_path(args.camera, RULE_ID)
            write_clip_from_buffer(out_path, ring.dump(), post_frames, fps_assumed)

            # guardar evento
            meta = {"motion_area_px": int(motion_area)}
            store.save_event(args.camera, RULE_ID, SEVERITY, str(out_path), meta)

            cooldown = fps_assumed * 2  # 2s sin volver a alertar

            # feedback visual
            cv2.putText(disp, "ALERTA: Movimiento en ROI", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        if cooldown > 0:
            cooldown -= 1
            cv2.putText(disp, f"cooldown: {cooldown/fps_assumed:.1f}s", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # mostrar máscara de depuración (opcional)
        small = cv2.resize(th, (w//4, h//4))
        disp[0:small.shape[0], w-small.shape[1]:w] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

        cv2.imshow("AI Camera Demo", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
