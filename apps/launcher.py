# apps/launcher.py
from __future__ import annotations
import os, sys, json, subprocess, shutil
from pathlib import Path
from typing import Dict, Any
import tkinter as tk
from tkinter import filedialog

# --- Paths base ---
REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
DATA = REPO / "data"
CAMERAS_YAML = CONFIGS / "cameras.yaml"
ROIS_DIR = CONFIGS / "rois"
LAST_SESSION = CONFIGS / "last_session.json"

# --- Defaults ---
DEFAULT_CAMERA_ID = "CAM_DEMO"
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 1280, 720, 25
DEFAULT_CONF_HAND = 0.5
DEFAULT_CONF_YOLO = 0.35
DEFAULT_YOLO_PATH = "yolov8n.pt"

# --- YAML helpers (no dependencias externas) ---
def load_yaml(path: Path) -> Dict[str, Any]:
    """Parser YAML minimalista para cameras.yaml (clave-valor, strings entre comillas)."""
    if not path.exists():
        return {}
    out: Dict[str, Any] = {}
    current_cam = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and not line.startswith(" "):  # top key
                key = line[:-1].strip()
                out[key] = {}
            elif line.endswith(":") and line.startswith("  "):   # camera id
                current_cam = line[:-1].strip()
                out.setdefault("cameras", {})
                out["cameras"].setdefault(current_cam, {})
            elif ":" in line and line.startswith("    "):
                k, v = [x.strip() for x in line.split(":", 1)]
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                else:
                    try:
                        if "." in v:
                            v = float(v)
                            if v.is_integer(): v = int(v)
                        else:
                            v = int(v)
                    except Exception:
                        pass
                if current_cam is not None:
                    out["cameras"][current_cam][k] = v
    return out

def dump_yaml_cameras(cameras: Dict[str, Dict[str, Any]]) -> str:
    lines = ["cameras:"]
    for cam_id, cfg in cameras.items():
        lines.append(f"  {cam_id}:")
        for k in ("source", "width", "height", "fps_target"):
            if k in cfg:
                v = cfg[k]
                if isinstance(v, str):
                    lines.append(f'    {k}: "{v}"')
                else:
                    lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"

def ensure_dirs():
    (REPO / "apps").mkdir(exist_ok=True)
    (REPO / "core").mkdir(exist_ok=True)
    CONFIGS.mkdir(exist_ok=True)
    ROIS_DIR.mkdir(exist_ok=True)
    (DATA / "clips").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "samples").mkdir(parents=True, exist_ok=True)
    (REPO / "outputs").mkdir(parents=True, exist_ok=True)

def save_last_session(data: Dict[str, Any]):
    LAST_SESSION.parent.mkdir(exist_ok=True)
    with LAST_SESSION.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_last_session() -> Dict[str, Any]:
    if LAST_SESSION.exists():
        try:
            return json.loads(LAST_SESSION.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def pick_video_file() -> str | None:
    root = tk.Tk(); root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [("Videos", "*.mp4;*.avi;*.mov;*.mkv;*.m4v"), ("All files", "*.*")]
    path = filedialog.askopenfilename(
        title="Selecciona un video",
        initialdir=str(DATA / "samples"),
        filetypes=filetypes
    )
    root.destroy()
    if not path: return None
    p = Path(path)
    try:
        rel = p.relative_to(REPO)
        return f"file://{str(rel).replace(os.sep, '/')}"
    except ValueError:
        return f"file:///{str(p).replace(os.sep, '/')}"

def update_camera_source(camera_id: str, source_url: str, width: int, height: int, fps: int):
    cfg = load_yaml(CAMERAS_YAML)
    cams = cfg.get("cameras", {})
    cams[camera_id] = {
        "source": source_url,
        "width": width,
        "height": height,
        "fps_target": fps,
    }
    CAMERAS_YAML.write_text(dump_yaml_cameras(cams), encoding="utf-8")
    return cams[camera_id]

def run_module(mod: str, args: list[str]) -> int:
    """Ejecuta un módulo Python en un subproceso con el entorno correcto."""
    cmd = [sys.executable, "-m", mod] + args
    print(">", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    print(f"[subprocess] return code: {rc}")
    return rc

def menu_input(prompt: str, default: str | None = None) -> str:
    s = input(f"{prompt} " + (f"[{default}]: " if default else ": "))
    if not s and default is not None:
        return default
    return s

# -------------------------------
# Sub-menú: Event Viewer (CLI)
# -------------------------------
def event_viewer_menu(last: Dict[str, Any]):
    print("\n--- Event Viewer ---")
    camera = menu_input("Filtro camera_id (vacío = todos)", last.get("viewer_camera", ""))
    rule   = menu_input("Filtro rule_id (vacío = todos)",   last.get("viewer_rule",   ""))
    dfrom  = menu_input("Fecha desde YYYY-mm-dd (vacío = sin límite)", last.get("viewer_from", ""))
    dto    = menu_input("Fecha hasta YYYY-mm-dd (vacío = sin límite)", last.get("viewer_to",   ""))

    # guarda filtros
    last.update({"viewer_camera": camera, "viewer_rule": rule, "viewer_from": dfrom, "viewer_to": dto})
    save_last_session(last)

    def common_args():
        args = []
        if camera: args += ["--camera", camera]
        if rule:   args += ["--rule",   rule]
        if dfrom:  args += ["--date_from", dfrom]
        if dto:    args += ["--date_to",   dto]
        return args

    while True:
        print("\nAcciones Event Viewer:")
        print("  1) Listar en consola (con filtros)")
        print("  2) Mostrar mosaico (--show)")
        print("  3) Abrir clip por índice (--open N)")
        print("  4) Exportar filtrado (CSV/XLSX)")
        print("  5) Volver")
        choice = menu_input("Elige opción", "1").strip()

        if choice == "1":
            run_module("apps.event_viewer", common_args())

        elif choice == "2":
            cols = menu_input("Columnas del mosaico", str(last.get("viewer_cols", 3)))
            try:
                last["viewer_cols"] = int(cols)
            except Exception:
                last["viewer_cols"] = 3
            save_last_session(last)
            run_module("apps.event_viewer", common_args() + ["--show", "--cols", str(last["viewer_cols"])])

        elif choice == "3":
            idx = menu_input("Índice del evento (según listado)", "0")
            run_module("apps.event_viewer", common_args() + ["--open", idx])

        elif choice == "4":
            # ruta por defecto
            default_out = last.get("viewer_export_path", str(REPO / "outputs" / "events_filtered.csv"))
            out_path = menu_input("Ruta de exportación (.csv o .xlsx)", default_out)
            last["viewer_export_path"] = out_path
            save_last_session(last)
            run_module("apps.event_viewer", common_args() + ["--export", out_path])

        else:
            break

# -------------------------------
# NUEVO: Modo Operador
# -------------------------------
def launch_operator_mode(last: Dict[str, Any]):
    """
    Abre el mosaico y el visor UI al mismo tiempo.
    - Si existe apps/multi_vision.py lo usa para el mosaico en vivo.
    - Si no existe, usa apps/event_viewer --show como fallback (mosaico por thumbnails de eventos).
    """
    # Definir columnas del mosaico
    cols = menu_input("Columnas del mosaico (2-4)", str(last.get("op_cols", 3)))
    try:
        cols_i = max(2, min(4, int(cols)))
    except Exception:
        cols_i = 3
    last["op_cols"] = cols_i
    save_last_session(last)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)

    # 1) Lanzar Event Viewer UI (incidencias)
    ui_cmd = [sys.executable, "-m", "apps.event_viewer_ui"]
    print(">", " ".join(ui_cmd))
    ui_proc = subprocess.Popen(ui_cmd, cwd=str(REPO), env=env)

    # 2) Lanzar mosaico (preferir multi_vision si existe)
    mv_path = REPO / "apps" / "multi_vision.py"
    if mv_path.exists():
        mosaic_cmd = [sys.executable, "-m", "apps.multi_vision", "--cols", str(cols_i)]
    else:
        # Fallback: usar event_viewer en modo grid (no es en vivo, pero funciona para operador básico)
        mosaic_cmd = [sys.executable, "-m", "apps.event_viewer", "--show", "--cols", str(cols_i)]
    print(">", " ".join(mosaic_cmd))
    mosaic_proc = subprocess.Popen(mosaic_cmd, cwd=str(REPO), env=env)

    print("\n[INFO] 🚦 Modo Operador iniciado.")
    print("      - Cierra las ventanas para terminar el modo.")
    print("      - Este modo no bloquea la consola; puedes minimizarla.")

def main():
    ensure_dirs()
    last = load_last_session()
    print("\n=== AI Camera Launcher ===")
    camera_id = menu_input("Camera ID a usar", last.get("camera_id", DEFAULT_CAMERA_ID))
    src_current = load_yaml(CAMERAS_YAML).get("cameras", {}).get(camera_id, {}).get("source", "")
    print(f"Fuente actual: {src_current or '(no definida)'}")
    print("\nAcciones:")
    print("  0) 🚦 Modo Operador (Mosaico + Incidencias)")
    print("  1) Elegir/Cambiar video… (actualiza cameras.yaml)")
    print("  2) Editar ROI de esta cámara")
    print("  3) Ejecutar detección: Hands (MediaPipe)")
    print("  4) Ejecutar detección: YOLO (personas)")
    print("  5) Ejecutar detección por reglas (rules.yaml)")
    print("  6) Event Viewer (listar/filtrar, mosaico, abrir clip, exportar)")
    print("  7) Event Viewer UI (visualizador con lista y doble-clic)")
    print("  8) Salir")

    choice = menu_input("Selecciona opción", "0").strip()

    if choice == "0":
        launch_operator_mode(last)

    elif choice == "1":
        path = pick_video_file()
        if not path:
            print("No se seleccionó video.")
            return
        w = int(menu_input("Width destino", str(last.get("width", DEFAULT_WIDTH))))
        h = int(menu_input("Height destino", str(last.get("height", DEFAULT_HEIGHT))))
        fps = int(menu_input("FPS destino", str(last.get("fps", DEFAULT_FPS))))
        cam_cfg = update_camera_source(camera_id, path, w, h, fps)
        print(f"[OK] cameras.yaml actualizado para {camera_id} → {cam_cfg['source']}")
        last.update({"camera_id": camera_id, "width": w, "height": h, "fps": fps})
        save_last_session(last)

        if menu_input("¿Abrir editor ROI ahora? (s/n)", "s").lower().startswith("s"):
            run_module("apps.tools.roi_editor", ["--camera", camera_id])

        if menu_input("¿Correr Hands ahora? (s/n)", "s").lower().startswith("s"):
            conf = menu_input("Conf Hands", str(last.get("conf_hand", DEFAULT_CONF_HAND)))
            last["conf_hand"] = float(conf)
            save_last_session(last)
            run_module("apps.vision_loop", ["--camera", camera_id, "--model", "hand", "--conf", conf])

    elif choice == "2":
        rc = run_module("apps.tools.roi_editor", ["--camera", camera_id])
        if rc != 0:
            print("ROI editor terminó con error.")

    elif choice == "3":
        conf = menu_input("Conf Hands", str(last.get("conf_hand", DEFAULT_CONF_HAND)))
        last["conf_hand"] = float(conf)
        save_last_session(last)
        run_module("apps.vision_loop", ["--camera", camera_id, "--model", "hand", "--conf", conf])

    elif choice == "4":
        conf = menu_input("Conf YOLO", str(last.get("conf_yolo", DEFAULT_CONF_YOLO)))
        ypath = menu_input("YOLO .pt", last.get("yolo_path", DEFAULT_YOLO_PATH))
        last["conf_yolo"] = float(conf)
        last["yolo_path"] = ypath
        save_last_session(last)
        run_module("apps.vision_loop", ["--camera", camera_id, "--model", "yolo", "--yolo_path", ypath, "--conf", conf])

    elif choice == "5":
        conf = menu_input("Conf default (sin override)", str(last.get("conf_rules", 0.35)))
        ypath = menu_input("YOLO .pt", last.get("yolo_path", DEFAULT_YOLO_PATH))
        last["conf_rules"] = float(conf)
        last["yolo_path"] = ypath
        save_last_session(last)
        run_module("apps.vision_rules", ["--camera", camera_id, "--conf", conf, "--yolo_path", ypath])

    elif choice == "6":
        event_viewer_menu(last)

    elif choice == "7":
        # Abre el visualizador con UI (lista + doble clic)
        run_module("apps.event_viewer_ui", [])

    else:
        print("Saliendo…")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
