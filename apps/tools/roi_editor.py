import argparse, cv2, numpy as np
from typing import List, Tuple
from core.io_utils import repo_path, load_yaml, save_roi_polygon, load_roi_polygons
from core.video_source import VideoSource, SourceConfig

WINDOW = "ROI Editor"

def load_camera_cfg(camera_id: str):
    cfg_path = repo_path("configs", "cameras.yaml")
    cfg = load_yaml(cfg_path); cameras = cfg.get("cameras", {})
    if camera_id not in cameras:
        raise RuntimeError(f"camera_id '{camera_id}' no encontrado")
    c = cameras[camera_id]
    return SourceConfig(url=c.get("source"), width=c.get("width"), height=c.get("height"), fps_target=c.get("fps_target"))

class ROIEditor:
    def __init__(self, frame_bgr):
        self.frame = frame_bgr.copy(); self.overlay = frame_bgr.copy()
        self.points: List[Tuple[int,int]] = []; self.closed = False
    def draw(self):
        display = self.overlay.copy()
        if self.points:
            for i,(x,y) in enumerate(self.points):
                cv2.circle(display,(x,y),4,(0,255,0),-1)
                if i>0: cv2.line(display, self.points[i-1], (x,y), (0,200,0), 2)
        if self.closed and len(self.points)>=3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(display, [pts], (0,0,255))
            display = cv2.addWeighted(self.frame, 0.6, display, 0.4, 0)
        help_text = "Izq: punto | Der: cerrar | S: guardar | R: reset | Q/Esc: salir"
        cv2.putText(display, help_text, (16,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return display
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            self.points.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN and not self.closed and len(self.points)>=3:
            self.closed = True

def grab_first_frame(src: VideoSource):
    if not src.open(): raise RuntimeError("No se pudo abrir la fuente")
    ok, frame = src.read()
    if not ok: raise RuntimeError("No se pudo leer frame")
    if src.cap: src.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    args = ap.parse_args()

    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg); frame = grab_first_frame(src)

    existing = load_roi_polygons(args.camera)
    editor = ROIEditor(frame)
    if existing: editor.points = existing[0]; editor.closed = len(editor.points)>=3

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, editor.on_mouse)

    while True:
        display = editor.draw(); cv2.imshow(WINDOW, display)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('r'): editor.points = []; editor.closed = False
        elif key == ord('s'):
            if editor.closed and len(editor.points)>=3:
                path = save_roi_polygon(args.camera, editor.points)
                print(f"[ROI] Guardado: {path}")
            else:
                print("[ROI] Cierra el polígono (click derecho, mínimo 3 puntos)")
    cv2.destroyAllWindows(); src.release()

if __name__ == "__main__":
    main()
