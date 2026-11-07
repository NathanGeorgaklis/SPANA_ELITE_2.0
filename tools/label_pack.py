"""Make an offline labelling pack for ONE DEV clip: sampled raw frames + manifest + the labelling page.

    python -m tools.label_pack --video CLIP.mp4 --split DEV --step-s 0.2 --out PACK_DIR

The pack holds raw frames only (never model output: truth comes before inference). It refuses any split other than
DEV, refuses to overwrite, and records the clip's sha256 so the frozen labels can be tied to the exact video.
Open PACK_DIR/index.html in Chrome (double-click); no server or internet needed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

from spana.video import VideoSource

PACK_SCHEMA = ("spana.label_pack", "1.0.0")
PAGE = Path(__file__).resolve().parent / "labeller.html"
JPEG_QUALITY = 92


def make_pack(video: str, split: str, step_s: float, out: str) -> dict:
    if split != "DEV":
        raise SystemExit(f"refusing split {split!r}: only DEV clips may be labelled now")
    if not step_s > 0:
        raise SystemExit("--step-s must be > 0")
    dest = Path(out)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    src = VideoSource(video)
    step = max(1, int(round(step_s * src.meta.fps)))
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "frames").mkdir(parents=True)
    frames = []
    for idx, t, img in src.frames():
        if idx % step:
            continue
        name = f"frames/{idx:06d}.jpg"
        cv2.imwrite(str(tmp / name), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frames.append({"index": idx, "time_s": round(t, 6), "file": name})
    if not frames:
        raise SystemExit("no frames decoded")
    manifest = {
        "schema": PACK_SCHEMA[0], "schema_version": PACK_SCHEMA[1], "split": split,
        "clip": {"name": Path(video).stem, "sha256": src.meta.sha256, "fps": src.meta.fps,
                 "width": src.meta.width, "height": src.meta.height},
        "frame_step": step, "frames": frames,
    }
    (tmp / "manifest.js").write_text("window.SPANA_PACK = " + json.dumps(manifest, indent=1) + ";\n")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(PAGE, tmp / "index.html")
    tmp.rename(dest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--step-s", type=float, default=0.2, help="seconds between labelled frames")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = make_pack(a.video, a.split, a.step_s, a.out)
    print(f"pack {a.out}: {len(m['frames'])} frames every {m['frame_step']} video frames ({m['clip']['name']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
