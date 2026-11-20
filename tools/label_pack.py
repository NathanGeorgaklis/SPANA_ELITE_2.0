"""Make an offline labelling pack for ONE DEV clip: sampled raw frames + manifest + the labelling page.

    python -m tools.label_pack --video CLIP.mp4 --split DEV --step-s 0.2 --out PACK_DIR
    python -m tools.label_pack --video MATCH.mp4 --split DEV --name NAME --start-frame A --end-frame B \
        (--step-s S | --step-frames N) --out PACK_DIR

The second form (pack 1.1.0) labels a named frame range [A, B) of a source video: frame indices stay those of the
source video and clip.sha256 is the source video's hash, so the frozen labels line up with tracks of that video;
`source` records the video name and range; the labels freeze under data/labels/NAME/. Without --name / range the pack
is the 1.0.0 whole-clip pack, unchanged.

Detector-assisted (pack 1.2.0, DECISIONS 2026-11-20, Nathan): with --suggestions DETECTIONS.json --suggest-min-score S
--cover-iou C every frame carries the detections of that spana.detections document (same video sha256, every pack frame
covered) scoring >= S, as SUGGESTIONS: the page shows them dashed; a box exists only once the annotator accepts or
draws it, and every ID is the annotator's. `assisted` records the detections file and the rule, and travels into the
frozen labels, so assisted truth is never mistaken for unassisted truth.

Without --suggestions the pack holds raw frames only (no model output: truth comes before inference). Every pack
refuses any split other than DEV, refuses to overwrite, and records the clip's sha256 so the frozen labels can be tied
to the exact video.
Open PACK_DIR/index.html in Chrome (double-click); no server or internet needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2

from spana.video import VideoSource

PACK_SCHEMA = ("spana.label_pack", "1.0.0")
RANGE_PACK_SCHEMA = ("spana.label_pack", "1.1.0")        # + source {video, start_frame, end_frame}; clip.name = NAME
ASSISTED_PACK_SCHEMA = ("spana.label_pack", "1.2.0")     # 1.1.0 + assisted {...}, frames[].suggestions
PAGE = Path(__file__).resolve().parent / "labeller.html"
JPEG_QUALITY = 92


def make_pack(video: str, split: str, step_s: Optional[float], out: str, name: Optional[str] = None,
              start_frame: Optional[int] = None, end_frame: Optional[int] = None,
              step_frames: Optional[int] = None, suggestions: Optional[str] = None,
              suggest_min_score: Optional[float] = None, cover_iou: Optional[float] = None) -> dict:
    if split != "DEV":
        raise SystemExit(f"refusing split {split!r}: only DEV clips may be labelled now")
    if (step_s is None) == (step_frames is None):
        raise SystemExit("give exactly one of --step-s / --step-frames")
    if step_s is not None and not step_s > 0:
        raise SystemExit("--step-s must be > 0")
    if step_frames is not None and not (isinstance(step_frames, int) and step_frames >= 1):
        raise SystemExit("--step-frames must be an integer >= 1")
    ranged = name is not None or start_frame is not None or end_frame is not None
    if ranged and (name is None or start_frame is None or end_frame is None):
        raise SystemExit("a range pack needs --name, --start-frame and --end-frame")
    if ranged and not (0 <= start_frame < end_frame):
        raise SystemExit("need 0 <= --start-frame < --end-frame")
    if ranged and not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise SystemExit("--name: letters, digits, _ and - only (it becomes a folder name)")
    if suggestions is not None and not ranged:
        raise SystemExit("--suggestions needs a range pack (--name / --start-frame / --end-frame)")
    if suggestions is not None and not (suggest_min_score is not None and 0 <= suggest_min_score <= 1
                                        and cover_iou is not None and 0 < cover_iou <= 1):
        raise SystemExit("--suggestions needs --suggest-min-score in [0, 1] and --cover-iou in (0, 1]")
    dest = Path(out)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    src = VideoSource(video)
    sugg_rows = None
    if suggestions is not None:
        from spana.artifacts import validate_detections
        raw = Path(suggestions).read_bytes()
        ddoc = json.loads(raw)
        validate_detections(ddoc)
        if ddoc["video"]["sha256"] != src.meta.sha256:
            raise SystemExit("--suggestions: detections of another video")
        sugg_rows = {fr["frame"]: fr["detections"] for fr in ddoc["frames"]}
        assisted = {"detections_file": Path(suggestions).name, "detections_sha256": hashlib.sha256(raw).hexdigest(),
                    "model": ddoc["model"], "min_score": suggest_min_score, "cover_iou": cover_iou,
                    "rule": "detector-assisted DEV truth: suggestions only; every box accepted or drawn and every ID "
                            "set by the annotator (DECISIONS 2026-11-20)"}
    step = step_frames if step_frames is not None else max(1, int(round(step_s * src.meta.fps)))
    first = start_frame if ranged else 0
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "frames").mkdir(parents=True)
    frames = []
    for idx, t, img in src.frames(first, end_frame if ranged else None):
        if (idx - first) % step:
            continue
        fname = f"frames/{idx:06d}.jpg"
        cv2.imwrite(str(tmp / fname), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        entry = {"index": idx, "time_s": round(t, 6), "file": fname}
        if sugg_rows is not None:
            if idx not in sugg_rows:
                raise SystemExit(f"--suggestions: frame {idx} has no detections (the run does not cover the pack)")
            entry["suggestions"] = [[round(float(r[0]), 1), round(float(r[1]), 1), round(float(r[2]), 1),
                                     round(float(r[3]), 1), round(float(r[4]), 3)]
                                    for r in sugg_rows[idx] if float(r[4]) >= suggest_min_score]
        frames.append(entry)
    if not frames:
        raise SystemExit("no frames decoded")
    if ranged and frames[-1]["index"] + step < end_frame:
        raise SystemExit(f"video ended at frame {frames[-1]['index']} before --end-frame {end_frame}")
    schema = ASSISTED_PACK_SCHEMA if sugg_rows is not None else (RANGE_PACK_SCHEMA if ranged else PACK_SCHEMA)
    manifest = {
        "schema": schema[0], "schema_version": schema[1], "split": split,
        "clip": {"name": name if ranged else Path(video).stem, "sha256": src.meta.sha256, "fps": src.meta.fps,
                 "width": src.meta.width, "height": src.meta.height},
        "frame_step": step, "frames": frames,
    }
    if ranged:
        manifest["source"] = {"video": Path(video).name, "start_frame": start_frame, "end_frame": end_frame}
    if sugg_rows is not None:
        manifest["assisted"] = assisted
    (tmp / "manifest.js").write_text("window.SPANA_PACK = " + json.dumps(manifest, indent=1) + ";\n")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(PAGE, tmp / "index.html")
    tmp.rename(dest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--step-s", type=float, default=None, help="seconds between labelled frames (default 0.2)")
    p.add_argument("--step-frames", type=int, default=None, help="video frames between labelled frames (1 = every)")
    p.add_argument("--name", default=None, help="range pack: label-set name (the frozen labels' folder)")
    p.add_argument("--start-frame", type=int, default=None, help="range pack: first video frame (inclusive)")
    p.add_argument("--end-frame", type=int, default=None, help="range pack: end video frame (exclusive)")
    p.add_argument("--suggestions", default=None, help="assisted pack: spana.detections of the same video")
    p.add_argument("--suggest-min-score", type=float, default=None, help="assisted pack: lowest detection score shown")
    p.add_argument("--cover-iou", type=float, default=None,
                   help="assisted pack: a suggestion is hidden once a label box overlaps it by this IoU")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    step_s = 0.2 if a.step_s is None and a.step_frames is None else a.step_s
    m = make_pack(a.video, a.split, step_s, a.out, a.name, a.start_frame, a.end_frame, a.step_frames, a.suggestions,
                  a.suggest_min_score, a.cover_iou)
    print(f"pack {a.out}: {len(m['frames'])} frames every {m['frame_step']} video frames ({m['clip']['name']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
