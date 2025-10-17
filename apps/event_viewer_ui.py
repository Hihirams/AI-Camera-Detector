import os
import sys
import json
import datetime as dt
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Asegura que se pueda importar el event_viewer base
HERE = Path(__file__).resolve()
APPS_DIR = HERE.parent
REPO_ROOT = Path(r"C:\Dev\AI-Camera-Detector")
sys.path.append(str(APPS_DIR))

try:
    import event_viewer as ev  # reutiliza funciones de lectura y preview
except Exception as e:
    raise SystemExit(f"[ERR] No se pudo importar event_viewer.py: {e}")

# 🔧 CONFIGURACIÓN FIJA: Apunta directamente a la ruta correcta
EVENTS_CSV = REPO_ROOT / "data" / "events.csv"

# Verificación de existencia al inicio
if not EVENTS_CSV.exists():
    print(f"[WARN] No se encontró events.csv en {EVENTS_CSV}")
    print(f"[INFO] Asegúrate de que el archivo existe en esa ubicación.")
else:
    print(f"[OK] events.csv encontrado en: {EVENTS_CSV}")

# ---------------------------------------------------------
# Interfaz gráfica principal
# ---------------------------------------------------------
class EventViewerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Camera — Event Viewer (Lista)")
        self.geometry("1000x600")
        self.minsize(900, 520)

        # ---- Filtros superiores ----
        f = ttk.Frame(self, padding=6)
        f.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(f, text="Camera:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.var_camera = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_camera, width=16).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(f, text="Rule:").grid(row=0, column=2, padx=4, pady=2, sticky="w")
        self.var_rule = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_rule, width=20).grid(row=0, column=3, padx=4, pady=2)

        ttk.Label(f, text="From (YYYY-mm-dd):").grid(row=0, column=4, padx=4, pady=2, sticky="w")
        self.var_from = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_from, width=14).grid(row=0, column=5, padx=4, pady=2)

        ttk.Label(f, text="To:").grid(row=0, column=6, padx=4, pady=2, sticky="w")
        self.var_to = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_to, width=14).grid(row=0, column=7, padx=4, pady=2)

        ttk.Button(f, text="Load / Filter", command=self.load_data).grid(row=0, column=8, padx=6, pady=2)
        ttk.Button(f, text="Export…", command=self.export_filtered).grid(row=0, column=9, padx=4, pady=2)

        # ---- Tabla principal (Treeview) ----
        cols = ("time", "camera", "rule", "duration_s", "detections_total", "clip_path")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("time", text="Time")
        self.tree.heading("camera", text="Camera")
        self.tree.heading("rule", text="Rule")
        self.tree.heading("duration_s", text="Dur (s)")
        self.tree.heading("detections_total", text="Detections")
        self.tree.heading("clip_path", text="Clip Path")

        # Tamaños de columnas
        self.tree.column("time", width=170, anchor="w")
        self.tree.column("camera", width=100, anchor="w")
        self.tree.column("rule", width=160, anchor="w")
        self.tree.column("duration_s", width=70, anchor="e")
        self.tree.column("detections_total", width=90, anchor="e")
        self.tree.column("clip_path", width=380, anchor="w")

        # Scrollbars
        ysb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ysb.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- Controles inferiores ----
        b = ttk.Frame(self, padding=6)
        b.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(b, text="Preview (OpenCV)", command=self.open_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Open in Default Player", command=self.open_external).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Refresh", command=self.load_data).pack(side=tk.RIGHT, padx=4)

        # Carga inicial
        self.rows = []
        self.load_data()

        # Doble clic = abrir preview
        self.tree.bind("<Double-1>", lambda e: self.open_preview())

    # ---------------------------------------------------------
    def load_data(self):
        camera = self.var_camera.get().strip() or None
        rule = self.var_rule.get().strip() or None
        date_from = self.var_from.get().strip() or None
        date_to = self.var_to.get().strip() or None

        rows = ev.read_events(EVENTS_CSV)
        rows = ev.filter_events(rows, camera, rule, date_from, date_to)
        ev.enrich(rows)
        self.rows = rows

        # Limpia y llena tabla
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, r in enumerate(rows):
            ts = r.get("_ts")
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            self.tree.insert("", "end", iid=str(i),
                             values=(
                                 ts_str,
                                 r.get("camera_id", ""),
                                 r.get("rule_id", ""),
                                 r.get("duration_s", 0),
                                 r.get("detections_total", 0),
                                 r.get("clip_path", ""),
                             ))

        self.title(f"AI Camera — Event Viewer (Lista)  |  {len(rows)} eventos")

    # ---------------------------------------------------------
    def _get_selected_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Selecciona un evento de la lista.")
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.rows):
            messagebox.showerror("Error", "Selección inválida.")
            return None
        return self.rows[idx]

    def open_preview(self):
        r = self._get_selected_row()
        if not r:
            return
        try:
            ev.preview_clip(r)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el preview:\n{e}")

    def open_external(self):
        r = self._get_selected_row()
        if not r:
            return
        clip = r.get("clip_path") or ""
        if not clip:
            messagebox.showwarning("Aviso", "Este evento no tiene clip_path.")
            return
        p = Path(clip)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            messagebox.showerror("Error", f"No se encontró el archivo:\n{p}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))  # Windows
            elif sys.platform == "darwin":
                os.system(f"open '{p}'")  # macOS
            else:
                os.system(f"xdg-open '{p}'")  # Linux
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el archivo:\n{e}")

    def export_filtered(self):
        if not self.rows:
            messagebox.showinfo("Info", "No hay eventos para exportar.")
            return
        filetypes = [("CSV", "*.csv"), ("Excel", "*.xlsx")]
        path = filedialog.asksaveasfilename(
            title="Exportar resultados",
            defaultextension=".csv",
            filetypes=filetypes,
            initialdir=str(REPO_ROOT / "outputs"),
            initialfile="events_filtered.csv",
        )
        if not path:
            return
        path = Path(path)
        try:
            ev.export_rows(self.rows, path)
            messagebox.showinfo("Listo", f"Exportado a:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar:\n{e}")


# ---------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------
def main():
    if not EVENTS_CSV.exists():
        print(f"[WARN] No se encontró events.csv en {EVENTS_CSV}")
    app = EventViewerUI()
    app.mainloop()


if __name__ == "__main__":
    main()