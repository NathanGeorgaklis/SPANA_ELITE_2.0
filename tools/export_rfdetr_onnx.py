"""Export trained RF-DETR weights (downloaded from Roboflow) to ONNX for local inference. Run ONCE per model, in a
SEPARATE environment: rfdetr needs Python >= 3.10 and PyTorch, which the engine's pinned environment does not have.

    <export-venv>/bin/python tools/export_rfdetr_onnx.py --weights models/rfdetr_v1/weights.pt \
        --variant medium --num-classes 4 --resolution 576 --out models/rfdetr_v1/rfdetr-medium-576.onnx

Prints a JSON line with the input / output names and shapes and the sha256 of the weights and of the ONNX file, for
the pinned model record. Weights and ONNX files are never committed (.gitignore: *.pt, *.onnx). The ONNX graph takes
one float32 image (1, 3, R, R) and returns `dets` (1, Q, 4: cx, cy, w, h normalised) and `labels` (1, Q, C+1 logits).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", required=True, choices=("nano", "small", "medium", "base", "large"))
    ap.add_argument("--num-classes", type=int, required=True, help="classes without the background slot")
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import onnx
    import rfdetr
    weights, out = Path(a.weights).resolve(), Path(a.out).resolve()
    if out.exists():
        print(f"ERROR: {out} exists; exports are never overwritten", file=sys.stderr)
        return 3
    cls = getattr(rfdetr, f"RFDETR{a.variant.capitalize()}")
    model = cls(pretrain_weights=str(weights), num_classes=a.num_classes, resolution=a.resolution)
    with tempfile.TemporaryDirectory() as td:
        made = Path(model.export(output_dir=td, verbose=False))
        if made.is_dir():
            made = next(made.glob("*.onnx"))
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(made), out)
    g = onnx.load(str(out)).graph
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
    print(json.dumps({"onnx": out.name, "onnx_sha256": sha256(out), "weights_sha256": sha256(weights),
                      "variant": a.variant, "num_classes": a.num_classes, "resolution": a.resolution,
                      "rfdetr": getattr(rfdetr, "__version__", None), "onnx_opset": 17,
                      "inputs": {i.name: dims(i) for i in g.input}, "outputs": {o.name: dims(o) for o in g.output}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
