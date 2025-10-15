# core/rules/engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set
from core.io_utils import load_yaml, repo_path, load_roi_polygons

@dataclass
class RuleCfg:
    id: str
    model: str       # "hand" | "yolo" | (futuro: "face")
    roi: str         # ej. "roi_1"
    severity: str    # "low" | "medium" | "high" | etc.
    conf_override: float | None = None

def load_rules(path: Path | None = None) -> List[RuleCfg]:
    path = path or repo_path("configs", "rules.yaml")
    doc = load_yaml(path)
    raw_rules = doc.get("rules", []) if isinstance(doc, dict) else []
    out: List[RuleCfg] = []
    for r in raw_rules:
        out.append(
            RuleCfg(
                id=r.get("id", "RULE").strip(),
                model=r.get("model", "hand").strip(),
                roi=r.get("roi", "roi_1").strip(),
                severity=str(r.get("severity", "medium")).lower(),
                conf_override=float(r["conf_override"]) if "conf_override" in r else None,
            )
        )
    return out

def required_models(rules: List[RuleCfg]) -> Set[str]:
    return {r.model for r in rules}

def get_roi_points_for_camera(camera_id: str, roi_id: str):
    """
    Devuelve puntos del ROI. Si tu loader retorna solo una lista de polígonos,
    tomamos el primero. Si soporta IDs, intenta mach con roi_id; si no, fallback.
    """
    polys = load_roi_polygons(camera_id) or []
    # Fallback: primer polígono
    if isinstance(polys, list) and polys:
        # polys[0] es lista de (x,y)
        return polys[0]
    return None
