import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import yaml

ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def roi_path(camera_id: str) -> Path:
    return repo_path("configs", "rois", f"{camera_id}.json")

def save_roi_polygon(camera_id: str, polygon_xy: List[Tuple[int, int]]) -> Path:
    path = roi_path(camera_id)
    data = {
        "camera_id": camera_id,
        "polygons": [{"id": "roi_1", "points": [[int(x), int(y)] for (x,y) in polygon_xy]}],
    }
    save_json(data, path); return path

def load_roi_polygons(camera_id: str):
    path = roi_path(camera_id)
    if not path.exists(): return []
    data = load_json(path)
    polys = []
    for poly in data.get("polygons", []):
        pts = poly.get("points", [])
        polys.append([(int(p[0]), int(p[1])) for p in pts])
    return polys
