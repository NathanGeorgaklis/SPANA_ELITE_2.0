"""Pure helpers for tools/modal_reid.py (no Modal import, so they are testable anywhere).

The Modal app only moves tools/reid_train.py onto a cloud GPU: every path the caller passes is relative to the
Modal volume and may not leave it, and the training command is the same CLI as on the Mac with --device cuda and a
--max-minutes budget that ends before the Modal timeout (the run is resumable, so a rerun continues).
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import List, Sequence

import yaml

SCHEMA = "spana.modal_reid_config"
KEYS = {"schema", "schema_version", "app_name", "volume_name", "gpu", "python_version", "timeout_minutes",
        "stop_margin_minutes", "packages"}


def load_modal_config(path) -> dict:
    """Fail closed: exact key set, positive training budget."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not a {SCHEMA}")
    if set(cfg) != KEYS:
        raise SystemExit(f"{path}: keys {sorted(set(cfg) ^ KEYS)} missing or unknown")
    if train_budget_minutes(cfg) <= 0:
        raise SystemExit(f"{path}: stop_margin_minutes leaves no training time")
    return cfg


def train_budget_minutes(cfg: dict) -> float:
    return float(cfg["timeout_minutes"]) - float(cfg["stop_margin_minutes"])


def volume_path(mount: str, rel: str) -> str:
    """mount/rel, refusing absolute paths and anything that climbs out of the volume."""
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path must be relative to the volume and stay inside it: {rel!r}")
    return str(PurePosixPath(mount) / p)


def train_argv(mount: str, crops: Sequence[str], backbone: str, run: str, train_config: str,
               max_minutes: float) -> List[str]:
    if not crops:
        raise ValueError("at least one crops folder is required")
    return (["python", "-m", "tools.reid_train", "train", "--crops", *[volume_path(mount, c) for c in crops],
             "--backbone", volume_path(mount, backbone), "--out", volume_path(mount, run),
             "--config", train_config, "--device", "cuda", "--max-minutes", str(max_minutes)])


def export_argv(mount: str, run: str, backbone: str, out: str) -> List[str]:
    return ["python", "-m", "tools.reid_train", "export", "--run", volume_path(mount, run),
            "--backbone", volume_path(mount, backbone), "--out", volume_path(mount, out)]
