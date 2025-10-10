"""
Editor simple de ROI:
- Lee configs/cameras.yaml para obtener un camera_id y su fuente
- Abre el primer frame (o el frame actual si es archivo)
- Permite dibujar un polígono con clicks izquierdos
- Click derecho para cerrar el polígono
- Tecla 'S' para guardar en configs/rois/<camera_id>.json
- Tecla 'R' para resetear el polígono
- Tecla 'Q' o ESC para salir
"""

import argparse
import cv2
import numpy as np
from typing import List, Tuple

from core.io_utils import repo_path, load_yaml, save_roi_polygon, load_roi_polygons
from core.video_source import VideoSource, SourceConfig


WINDOW = "ROI Editor"


def load_camera_cfg(camera_id: str):
    cfg_path = repo_path("configs", "cameras.yaml")
    cfg = load_yaml(cfg_path)
    cameras = cfg.get("cameras", {})
    if camera_id not in cameras:
        raise RuntimeError(f"camera_id '{camera_id}' no encontrado en cameras.yaml")
    c = cameras[camera_id]
    return SourceConfig(
        url=c.get("source"),
        width=c.get("width"),
        height=c.get("height"),
        fps_target=c.get("fps_target"),
    )


class ROIEditor:
    def __init__(self, frame_bgr):
        self.frame = frame_bgr.copy()
        self.overlay = frame_bgr.copy()
        self.points: List[Tuple[int, int]] = []
        self.closed = False

    def draw(self):
        display = self.overlay.copy()

        # dibujar puntos y líneas
        if self.points:
            for i, (x, y) in enumerate(self.points):
                cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i - 1], (x, y), (0, 200, 0), 2)

        # si está cerrado, conectar último con primero y rellenar suavemente
        if self.closed and len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(display, [pts], (0, 0, 255))  # rojo sólido
            # hacerlo semitransparente sobre el frame
            display = cv2.addWeighted(self.frame, 0.6, display, 0.4, 0)

        # textos
        help_text = "Click izq: agregar punto | Click der: cerrar | S: guardar | R: reset | Q/Esc: salir"
        cv2.putText(display, help_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and not self.closed:
            if len(self.points) >= 3:
                self.closed = True


def grab_first_frame(src: VideoSource):
    """Obtiene un frame de referencia. Si es archivo, usa el primer frame."""
    if not src.open():
        raise RuntimeError("No se pudo abrir la fuente de video.")
    ok, frame = src.read()
    if not ok:
        raise RuntimeError("No se pudo leer un frame de la fuente.")
    # Para archivos, intentemos rebobinar al inicio
    if src.cap and isinstance(src.cap, cv2.VideoCapture):
        src.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="ID de cámara en configs/cameras.yaml (ej: CAM_DEMO)")
    args = ap.parse_args()

    # cargar fuente
    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg)
    frame = grab_first_frame(src)

    # si ya existe ROI, precargarlo
    existing = load_roi_polygons(args.camera)
    editor = ROIEditor(frame)
    if existing:
        # cargamos el primero
        editor.points = existing[0]
        editor.closed = len(editor.points) >= 3

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, editor.on_mouse)

    while True:
        display = editor.draw()
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord('q'), 27):  # q o ESC
            break
        elif key == ord('r'):
            editor.points = []
            editor.closed = False
        elif key == ord('s'):
            if editor.closed and len(editor.points) >= 3:
                path = save_roi_polygon(args.camera, editor.points)
                print(f"[ROI] Guardado: {path}")
            else:
                print("[ROI] Cierra el polígono con click derecho (mínimo 3 puntos).")

    cv2.destroyAllWindows()
    src.release()


if __name__ == "__main__":
    main()
