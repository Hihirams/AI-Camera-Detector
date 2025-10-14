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
        fps_target=c.get("fps_target")
    )


class ROIEditor:
    def __init__(self, frame_bgr):
        self.frame = frame_bgr.copy()
        self.overlay = frame_bgr.copy()
        self.points: List[Tuple[int, int]] = []
        self.closed = False
        print("[ROI Editor] Inicializado")
        print("[ROI Editor] Haz click IZQUIERDO para agregar puntos")

    def draw(self):
        display = self.frame.copy()

        # Dibujar puntos y líneas
        if self.points:
            for i, (x, y) in enumerate(self.points):
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 6, (255, 255, 255), 1)
                if i > 0:
                    cv2.line(display, self.points[i - 1], (x, y), (0, 255, 0), 2)

        # Si está cerrado, conectar último con primero y rellenar
        if self.closed and len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.line(display, self.points[-1], self.points[0], (0, 255, 0), 2)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # Textos de ayuda
        help_text = "Click IZQ: punto | Click DER: cerrar | S: guardar | R: reset | Q: salir"
        cv2.putText(display, help_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, help_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        status = f"Puntos: {len(self.points)} | Cerrado: {'SI' if self.closed else 'NO'}"
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return display

    def handle_mouse(self, event, x, y, flags, param):
        """Handler del mouse - será llamado por la función global"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.closed:
                self.points.append((x, y))
                print(f"[ROI] Punto {len(self.points)} agregado: ({x}, {y})")
            else:
                print("[ROI] Polígono cerrado. Usa 'R' para resetear")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.closed and len(self.points) >= 3:
                self.closed = True
                print(f"[ROI] Polígono CERRADO con {len(self.points)} puntos")
            elif len(self.points) < 3:
                print("[ROI] Necesitas al menos 3 puntos para cerrar")


def grab_first_frame(src: VideoSource):
    """Obtiene el primer frame de la fuente."""
    if not src.open():
        raise RuntimeError("No se pudo abrir la fuente de video")
    ok, frame = src.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer un frame de la fuente")
    # Rebobinar si es archivo
    if src.cap and isinstance(src.cap, cv2.VideoCapture):
        src.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="ID de cámara en configs/cameras.yaml")
    args = ap.parse_args()

    print(f"\n{'='*50}")
    print(f"ROI Editor - Camera: {args.camera}")
    print(f"{'='*50}\n")
    
    # Cargar fuente
    scfg = load_camera_cfg(args.camera)
    src = VideoSource(scfg)
    frame = grab_first_frame(src)
    print(f"✓ Frame cargado: {frame.shape[1]}x{frame.shape[0]}")

    # Cargar ROI existente si hay
    existing = load_roi_polygons(args.camera)
    editor = ROIEditor(frame)
    if existing:
        editor.points = existing[0]
        editor.closed = len(editor.points) >= 3
        print(f"✓ ROI existente cargado: {len(editor.points)} puntos")

    # Crear ventana
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    
    # CRÍTICO: Usar una función lambda que capture 'editor'
    def mouse_wrapper(event, x, y, flags, param):
        editor.handle_mouse(event, x, y, flags, param)
    
    cv2.setMouseCallback(WINDOW, mouse_wrapper)
    
    print("\nControles:")
    print("  • Click IZQUIERDO  → Agregar punto")
    print("  • Click DERECHO    → Cerrar polígono (min 3 puntos)")
    print("  • Tecla 'S'        → Guardar")
    print("  • Tecla 'R'        → Resetear")
    print("  • Tecla 'Q' o ESC  → Salir")
    print("\n¡Ventana abierta! Empieza a dibujar...\n")

    # Loop principal
    while True:
        display = editor.draw()
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:  # Q o ESC
            print("\n[ROI] Saliendo...")
            break
        
        elif key == ord('r') or key == ord('R'):
            editor.points = []
            editor.closed = False
            print("[ROI] ⟳ Polígono reseteado")
        
        elif key == ord('s') or key == ord('S'):
            if editor.closed and len(editor.points) >= 3:
                path = save_roi_polygon(args.camera, editor.points)
                print(f"[ROI] ✓ GUARDADO: {path}")
            else:
                print("[ROI] ✗ ERROR: Debes cerrar el polígono primero")
                print("[ROI]   (Click derecho con mínimo 3 puntos)")

    cv2.destroyAllWindows()
    src.release()
    print("\n[ROI] Editor cerrado\n")


if __name__ == "__main__":
    main()