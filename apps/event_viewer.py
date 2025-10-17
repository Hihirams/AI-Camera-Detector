import os, json, csv, argparse, datetime as dt
from pathlib import Path

# Opcional: pandas para exportar a Excel
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

import cv2
import numpy as np

# Carpeta raíz del proyecto
REPO_ROOT = Path(__file__).resolve().parents[1]

EVENTS_CSV = REPO_ROOT / "events.csv"
THUMBS_DIR = REPO_ROOT / "outputs" / "thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------
def parse_date(s: str) -> dt.datetime:
    """Intenta convertir diferentes formatos de fecha."""
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "").replace("T", " "))
    except Exception:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return dt.datetime.strptime(s, fmt)
            except Exception:
                pass
    try:
        return dt.datetime.fromtimestamp(float(s))
    except Exception:
        raise ValueError(f"Unrecognized date: {s}")


def read_events(path: Path) -> list[dict]:
    """Lee el CSV de eventos y normaliza campos."""
    if not path.exists():
        print(f"[WARN] No events.csv found at {path}")
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            try:
                r["_ts"] = parse_date(
    r.get("timestamp") or r.get("time") or r.get("ts") or r.get("ts_utc") or r.get("ts_iso_utc") or ""
)
            except Exception:
                r["_ts"] = None
            meta = r.get("meta") or "{}"
            try:
                r["_meta"] = json.loads(meta)
            except Exception:
                r["_meta"] = {}
            rows.append(r)
    return rows


def get_clip_info(clip_path: str) -> tuple[float, int, int]:
    """Devuelve (duración_s, total_frames, fps)."""
    if not clip_path:
        return (0.0, 0, 0)
    p = Path(clip_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        return (0.0, 0, 0)
    cap = cv2.VideoCapture(str(p))
    if not cap or not cap.isOpened():
        return (0.0, 0, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(cnt / fps) if fps and cnt else 0.0
    cap.release()
    return (duration, cnt, int(fps))


def ensure_thumb(clip_path: str) -> Path | None:
    """Crea miniatura del primer frame del clip."""
    if not clip_path:
        return None
    p = Path(clip_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        return None
    out = THUMBS_DIR / (p.stem + ".jpg")
    if out.exists():
        return out
    cap = cv2.VideoCapture(str(p))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    cv2.putText(frame, p.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out), frame)
    return out


def filter_events(rows: list[dict], camera: str | None, rule: str | None,
                  date_from: str | None, date_to: str | None) -> list[dict]:
    """Filtra por cámara, regla y rango de fechas."""
    df = parse_date(date_from) if date_from else None
    dt_ = parse_date(date_to) if date_to else None
    out = []
    for r in rows:
        if camera and (r.get("camera_id") or "") != camera:
            continue
        if rule and (r.get("rule_id") or "") != rule:
            continue
        if df and r.get("_ts") and r["_ts"] < df:
            continue
        if dt_ and r.get("_ts") and r["_ts"] > dt_:
            continue
        out.append(r)
    return out


def enrich(rows: list[dict]) -> None:
    """Agrega duración, total de detecciones y miniaturas."""
    for r in rows:
        clip_path = r.get("clip_path") or r.get("clip") or ""
        duration, frames, fps = get_clip_info(clip_path)
        r["duration_s"] = round(duration, 2)
        r["frames"] = frames
        r["fps"] = fps
        meta = r.get("_meta") or {}
        det = meta.get("detections") or meta.get("count") or meta.get("motion_area_px")
        try:
            r["detections_total"] = int(det)
        except Exception:
            r["detections_total"] = 0
        thumb = ensure_thumb(clip_path)
        r["thumbnail"] = str(thumb) if thumb else ""


def export_rows(rows: list[dict], out_path: Path) -> Path:
    """Exporta resultados a CSV o Excel."""
    out_path = out_path if out_path.is_absolute() else (REPO_ROOT / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".xlsx":
        if pd is None:
            csv_path = out_path.with_suffix(".csv")
            keys = rows[0].keys() if rows else []
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"[WARN] pandas/openpyxl no disponibles. Se guardó CSV → {csv_path}")
            return csv_path
        else:
            df = pd.DataFrame(rows)
            df.to_excel(out_path, index=False)
            return out_path
    else:
        keys = rows[0].keys() if rows else []
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return out_path


def draw_grid(rows: list[dict], cols: int = 4, cell_w: int = 320, cell_h: int = 180) -> np.ndarray:
    """Crea mosaico con miniaturas y texto."""
    if not rows:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    rows_n = int(np.ceil(len(rows) / cols))
    canvas = np.zeros((rows_n * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, r in enumerate(rows):
        thumb_path = r.get("thumbnail", "")
        if thumb_path and Path(thumb_path).exists():
            img = cv2.imread(thumb_path)
            if img is None:
                img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        else:
            img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        img = cv2.resize(img, (cell_w, cell_h))
        label = f"#{idx} {r.get('camera_id','?')} • {r.get('rule_id','?')}"
        info = f"{r.get('duration_s',0)}s • det:{r.get('detections_total',0)}"
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, info, (8, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        r0 = (idx // cols) * cell_h
        c0 = (idx % cols) * cell_w
        canvas[r0:r0 + cell_h, c0:c0 + cell_w] = img
    return canvas


def preview_clip(r: dict) -> None:
    """Abre un clip específico con overlay básico."""
    clip_path = r.get("clip_path") or r.get("clip") or ""
    if not clip_path:
        print("[ERR] No clip_path para este evento.")
        return
    p = Path(clip_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        print(f"[ERR] Clip no encontrado: {p}")
        return
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print("[ERR] No se pudo abrir el video.")
        return
    win = "Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cv2.putText(frame, f"{r.get('camera_id','')} • {r.get('rule_id','')}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            break
    cap.release()
    cv2.destroyWindow(win)


# -----------------------------------------------------
# Entrada principal
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AI Camera - Advanced Event Viewer")
    ap.add_argument("--camera", help="Filtrar por camera_id")
    ap.add_argument("--rule", help="Filtrar por rule_id")
    ap.add_argument("--date_from", help="Fecha inicial YYYY-mm-dd")
    ap.add_argument("--date_to", help="Fecha final YYYY-mm-dd")
    ap.add_argument("--export", help="Exportar a CSV/XLSX")
    ap.add_argument("--show", action="store_true", help="Mostrar mosaico de miniaturas")
    ap.add_argument("--open", type=int, help="Abrir clip por índice")
    ap.add_argument("--cols", type=int, default=4, help="Columnas del mosaico")
    args = ap.parse_args()

    rows = read_events(EVENTS_CSV)
    if not rows:
        print("[INFO] No hay eventos registrados.")
        return

    rows = filter_events(rows, args.camera, args.rule, args.date_from, args.date_to)
    enrich(rows)

    def short(p, n=28):
        p = str(p or "")
        return p if len(p) <= n else p[:n - 3] + "..."

    print(f"\nFiltered events: {len(rows)}")
    print(f"{'#':>3}  {'time':19}  {'camera':10}  {'rule':16}  {'dur_s':>6}  {'det':>3}  {'clip'}")
    for i, r in enumerate(rows):
        ts = r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else ""
        print(f"{i:>3}  {ts:19}  {short(r.get('camera_id'),10):10}  {short(r.get('rule_id'),16):16}  "
              f"{r.get('duration_s',0):6}  {r.get('detections_total',0):3}  {short(r.get('clip_path'),48)}")

    if args.export:
        out = export_rows(rows, Path(args.export))
        print(f"[OK] Exportado → {out}")

    if args.show:
        grid = draw_grid(rows, cols=args.cols)
        win = "Event Viewer"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, grid)
        print("\n[UI] Mostrando mosaico. Presiona ESC/Q para cerrar.")
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (27, ord('q')):
                break
        cv2.destroyWindow(win)

    if args.open is not None:
        if 0 <= args.open < len(rows):
            preview_clip(rows[args.open])
        else:
            print(f"[ERR] Índice fuera de rango (0..{len(rows)-1}).")


if __name__ == "__main__":
    main()
