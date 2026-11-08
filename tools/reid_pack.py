"""Make an offline re-ID labelling pack for ONE match: tracker-proposed tracklets + thumbnails + the labeller.

    python -m tools.reid_pack --video MATCH.mp4 --tracks MATCH.trk.json --manifest MATCH.reid_match.json \
        --out PACK_DIR [--thumbs 8]

The match manifest (split TRAIN/DEV/EVAL, footage licence, rights holder, consent basis, training_permitted) is
required and validated first (spana.reid_labels.validate_manifest). The tracks artifact must come from the same
video. Tracklets are the tracker's OBSERVED runs of each track, cut where the track was unobserved for longer than
--max-gap-s and where it entered IDENTITY_UNCERTAIN (a possible swap point), so a human never has to split at a
place the tracker already flagged. Boxes are the detector boxes the tracker matched: no box is drawn by hand.
Writes PACK_DIR/{index.html, pack.js, pack.json, tracklets.json, thumbs/}; never overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2

from spana.artifacts import read_json, validate_tracks
from spana.reid_labels import ReidLabelError, validate_manifest
from spana.video import VideoSource

PACK_SCHEMA = ("spana.reid_pack", "1.0.0")
PAGE = Path(__file__).resolve().parent / "reid_labeller.html"
THUMB_H = 112
MARGIN = 0.15


def propose(tracks_doc: dict, fps: float, max_gap_s: float, min_observed: int) -> List[dict]:
    """Tracklet proposals from a tracks artifact (OBSERVED frames only)."""
    runs: Dict[str, dict] = {}
    out: List[dict] = []
    seq: Dict[str, int] = {}

    def close(tid):
        r = runs.pop(tid, None)
        if r and len(r["frames"]) >= min_observed:
            out.append(r)

    for fr in tracks_doc["frames"]:
        f = fr["frame"]
        for t in fr["tracks"]:
            if t["source"] != "OBSERVED":
                continue
            tid = t["track_id"]
            r = runs.get(tid)
            new_episode = r is not None and t["uncertain_episodes"] > r["episodes"]
            gap = r is not None and (f - r["frames"][-1]) / fps > max_gap_s
            if r is None or new_episode or gap:
                close(tid)
                n = seq[tid] = seq.get(tid, 0) + 1
                r = runs[tid] = {"id": f"{tid}-{n}", "track": tid, "frames": [], "boxes": [],
                                 "episodes": t["uncertain_episodes"], "starts_uncertain": new_episode,
                                 "after_gap": bool(gap), "uncertain_frames": 0, "confirmed": False}
            r["frames"].append(f)
            r["boxes"].append([round(v, 1) for v in t["box"]])
            r["uncertain_frames"] += t["identity"] != "CERTAIN"
            r["confirmed"] |= t["confirmed"]
        for tid, _ in fr["ended"]:
            close(tid)
    for tid in list(runs):
        close(tid)
    out.sort(key=lambda r: (r["frames"][0], r["id"]))
    for r in out:
        r["first_frame"], r["last_frame"] = r["frames"][0], r["frames"][-1]
        del r["episodes"]
    return out


def make_pack(video: str, tracks: str, manifest_path: str, out: str, thumbs: int = 8, max_gap_s: float = 0.5,
              min_observed: int = 3) -> dict:
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    dest = Path(out)
    if dest.exists():
        raise ReidLabelError(f"refusing to overwrite {dest}")
    tdoc = read_json(tracks)
    validate_tracks(tdoc)
    src = VideoSource(video, expected_sha256=manifest["video_sha256"])
    if tdoc["video"]["sha256"] != manifest["video_sha256"]:
        raise ReidLabelError("the tracks artifact was made from a different video than the match manifest")
    tl = propose(tdoc, src.meta.fps, max_gap_s, min_observed)
    if not tl:
        raise ReidLabelError("no tracklets proposed")
    want: Dict[int, List[tuple]] = {}
    for r in tl:
        n = len(r["frames"])
        picks = sorted({round(k * (n - 1) / max(1, thumbs - 1)) for k in range(min(thumbs, n))})
        r["thumbs"] = []
        for k, p in enumerate(picks):
            r["thumbs"].append({"frame": r["frames"][p], "file": f"thumbs/{r['id']}_{k}.jpg"})
            want.setdefault(r["frames"][p], []).append((r["boxes"][p], r["thumbs"][-1]["file"]))
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "thumbs").mkdir(parents=True)
    last = max(want)
    for idx, _, img in src.frames(0, last + 1):
        for box, name in want.get(idx, ()):
            x1, y1, x2, y2 = box
            mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
            c = img[int(max(0, y1 - mh)):int(min(img.shape[0], y2 + mh)), int(max(0, x1 - mw)):int(min(img.shape[1], x2 + mw))]
            if c.size == 0:
                continue
            c = cv2.resize(c, (max(1, round(c.shape[1] * THUMB_H / c.shape[0])), THUMB_H))
            cv2.imwrite(str(tmp / name), c, [cv2.IMWRITE_JPEG_QUALITY, 85])
    tl_bytes = json.dumps(tl, separators=(",", ":")).encode()
    (tmp / "tracklets.json").write_bytes(tl_bytes)
    pack = {"schema": PACK_SCHEMA[0], "schema_version": PACK_SCHEMA[1], "match": manifest,
            "video": {"name": Path(video).name, "sha256": src.meta.sha256, "fps": src.meta.fps,
                      "width": src.meta.width, "height": src.meta.height},
            "tracks_sha256": hashlib.sha256(Path(tracks).read_bytes()).hexdigest(),
            "tracklets_sha256": hashlib.sha256(tl_bytes).hexdigest(),
            "proposal": {"max_gap_s": max_gap_s, "min_observed": min_observed, "thumbs": thumbs},
            "tracklets": [{k: r[k] for k in ("id", "track", "first_frame", "last_frame", "starts_uncertain",
                                             "after_gap", "uncertain_frames", "confirmed", "thumbs", "frames")}
                          | {"observed": len(r["frames"])} for r in tl]}
    (tmp / "pack.json").write_text(json.dumps(pack, indent=1))
    (tmp / "pack.js").write_text("window.SPANA_REID_PACK = " + json.dumps(pack) + ";\n")
    shutil.copy(PAGE, tmp / "index.html")
    tmp.rename(dest)
    return pack


def main() -> int:
    p = argparse.ArgumentParser()
    for k in ("video", "tracks", "manifest", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--thumbs", type=int, default=8)
    p.add_argument("--max-gap-s", type=float, default=0.5)
    p.add_argument("--min-observed", type=int, default=3)
    a = p.parse_args()
    try:
        pack = make_pack(a.video, a.tracks, a.manifest, a.out, a.thumbs, a.max_gap_s, a.min_observed)
    except ReidLabelError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"pack -> {a.out}: {len(pack['tracklets'])} tracklets, split {pack['match']['split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
