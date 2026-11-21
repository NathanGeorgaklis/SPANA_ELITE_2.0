"""Run tools/reid_train.py on a Modal cloud GPU (settings: config/modal_reid.yaml; training: config/reid_train.yaml).

Data lives on the Modal volume `volume_name`; upload it once from the repo root on the Mac:
    python3 -m modal volume put spana-reid CROPS_DIR crops/<name>
    python3 -m modal volume put spana-reid HF_DINOV2_SMALL_DIR backbones/dinov2_small
Train (resumable: rerun the same command to continue after a stop), then export the finished run:
    python3 -m modal run -m tools.modal_reid --crops crops/<a>,crops/<b> --backbone backbones/dinov2_small --run runs/<name>
    python3 -m modal run -m tools.modal_reid --run runs/<name> --backbone backbones/dinov2_small --export-to models/<name>.onnx
    python3 -m modal volume get spana-reid models/<name>.onnx .
All paths are relative to the volume. tools/reid_train.py keeps its own guards (TRAIN crops only, pinned backbone).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

from tools.modal_reid_plan import export_argv, load_modal_config, train_argv, train_budget_minutes, volume_path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root"  # tools/, spana/, config/, models/ sit here so tools.reid_train finds models/dinov2_small.json
MOUNT = "/vol"
CFG = load_modal_config(ROOT / "config" / "modal_reid.yaml")  # ROOT is REMOTE inside the container

image = (modal.Image.debian_slim(python_version=CFG["python_version"])
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install(*CFG["packages"])
         .add_local_python_source("tools", "spana")
         .add_local_dir(ROOT / "config", remote_path=f"{REMOTE}/config")
         .add_local_dir(ROOT / "models", remote_path=f"{REMOTE}/models"))
volume = modal.Volume.from_name(CFG["volume_name"], create_if_missing=True)
app = modal.App(CFG["app_name"], image=image)


def _run(argv) -> None:
    print(" ".join(argv), flush=True)
    try:
        subprocess.run(argv, cwd=REMOTE, check=True)
    finally:
        volume.commit()  # keep the checkpoint even when training stops with an error


@app.function(gpu=CFG["gpu"], timeout=int(CFG["timeout_minutes"]) * 60, volumes={MOUNT: volume})
def train(crops: list[str], backbone: str, run: str) -> None:
    volume.reload()
    _run(train_argv(MOUNT, crops, backbone, run, f"{REMOTE}/config/reid_train.yaml", train_budget_minutes(CFG)))


@app.function(gpu=CFG["gpu"], timeout=int(CFG["timeout_minutes"]) * 60, volumes={MOUNT: volume})
def export(run: str, backbone: str, out: str) -> None:
    volume.reload()
    Path(volume_path(MOUNT, out)).parent.mkdir(parents=True, exist_ok=True)  # the CLI cannot make folders
    _run(export_argv(MOUNT, run, backbone, out))


@app.local_entrypoint()
def main(run: str, backbone: str, crops: str = "", export_to: str = "") -> None:
    if export_to:
        export.remote(run, backbone, export_to)
    else:
        train.remote([c for c in crops.split(",") if c], backbone, run)
