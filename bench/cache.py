"""Frozen tracker input: the detector's output for one DEV clip, cached once, read identically by every tracker.

A cache dir holds
- `detections.json`: the `spana.detections` artifact made once by `spana.cli detect` (byte copy);
- `embeddings.npz` (optional): one SPANA re-ID v2 embedding per detection row, frame by frame (`emb`, `offsets`);
- `manifest.json`: sha256 of both files, the clip's video sha256 and the embedder that made the embeddings.
`DetectionCache.load` re-hashes both files and refuses any mismatch, so every candidate sees the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from spana.artifacts import detections_from_frame, read_json, validate_detections
from spana.records import Detection
from spana.video import sha256_file

CACHE_SCHEMA = ("spana.tracker_bench_cache", "1.0.0")


@dataclass(frozen=True)
class CachedFrame:
    index: int
    t: float
    detections: List[Detection]
    embeddings: Optional[np.ndarray]          # (len(detections), D) or None


class DetectionCache:
    def __init__(self, root: Path, manifest: dict, doc: dict, frames: List[CachedFrame]):
        self.root, self.manifest, self.doc, self.frames = root, manifest, doc, frames

    @property
    def video_sha256(self) -> str:
        return self.manifest["video_sha256"]

    @property
    def fps(self) -> float:
        return float(self.doc["video"]["fps"])

    @property
    def size(self) -> tuple:
        return int(self.doc["video"]["width"]), int(self.doc["video"]["height"])

    @property
    def has_embeddings(self) -> bool:
        return self.manifest["embeddings_sha256"] is not None

    def fingerprint(self) -> str:
        """sha256 over exactly what a tracker receives (frames, times, boxes, scores, classes, flags, embeddings)."""
        h = hashlib.sha256()
        for fr in self.frames:
            h.update(np.array([fr.index, fr.t], float).tobytes())
            for d in fr.detections:
                h.update(np.array([*d.box, d.score, d.class_id, d.multi_body], float).tobytes())
            if fr.embeddings is not None:
                h.update(np.ascontiguousarray(fr.embeddings, dtype=np.float32).tobytes())
        return h.hexdigest()

    @staticmethod
    def load(root: str | Path) -> "DetectionCache":
        root = Path(root)
        m = json.loads((root / "manifest.json").read_text())
        if (m.get("schema"), m.get("schema_version")) != CACHE_SCHEMA:
            raise ValueError(f"{root}: not a {CACHE_SCHEMA[0]} {CACHE_SCHEMA[1]} cache")
        if sha256_file(root / "detections.json") != m["detections_sha256"]:
            raise ValueError(f"{root}: detections.json does not match its manifest hash")
        doc = read_json(root / "detections.json")
        validate_detections(doc)
        if doc["video"]["sha256"] != m["video_sha256"]:
            raise ValueError(f"{root}: detections were made from another video")
        emb_rows = offsets = None
        if m["embeddings_sha256"] is not None:
            if sha256_file(root / "embeddings.npz") != m["embeddings_sha256"]:
                raise ValueError(f"{root}: embeddings.npz does not match its manifest hash")
            with np.load(root / "embeddings.npz") as z:
                emb_rows, offsets = z["emb"], z["offsets"]
            if len(offsets) != len(doc["frames"]) + 1 or offsets[-1] != len(emb_rows):
                raise ValueError(f"{root}: embeddings do not line up with the detection frames")
        frames = []
        for k, fr in enumerate(doc["frames"]):
            dets = detections_from_frame(fr)
            emb = None
            if emb_rows is not None:
                emb = emb_rows[offsets[k]:offsets[k + 1]]
                if len(emb) != len(dets):
                    raise ValueError(f"{root}: frame {fr['frame']}: {len(emb)} embeddings for {len(dets)} detections")
            frames.append(CachedFrame(int(fr["frame"]), float(fr["t"]), dets, emb))
        return DetectionCache(root, m, doc, frames)


def build_cache(out: str | Path, clip: str, detections_path: str | Path, labels_clip_sha256: str,
                video_path: Optional[str | Path] = None, embedder=None) -> Path:
    """Copy the detector's artifact into a new cache dir and, with an embedder, embed every detection once.
    Refuses an existing dir and detections made from a video other than the labelled clip's."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite cache {out}")
    doc = read_json(detections_path)
    validate_detections(doc)
    if doc["video"]["sha256"] != labels_clip_sha256:
        raise SystemExit(f"{clip}: detections were made from a video other than the labelled clip")
    emb_sha, emb_info = None, None
    rows: List[np.ndarray] = []
    offsets = [0]
    if embedder is not None:
        from spana.identity.reid import embed_boxes
        from spana.video import VideoSource
        if video_path is None:
            raise SystemExit("embedding needs --video")
        video = VideoSource(video_path, expected_sha256=labels_clip_sha256)
        want = {fr["frame"]: fr for fr in doc["frames"]}
        first, last = min(want), max(want)
        t0, done = time.perf_counter(), []
        for i, _, img in video.frames(first, last + 1):
            if i not in want:
                continue
            boxes = [r[:4] for r in want[i]["detections"]]
            embs = embed_boxes(embedder, img, boxes)
            if any(e is None for e in embs):
                raise SystemExit(f"{clip} frame {i}: empty crop; a detection box must lie inside the frame")
            rows.extend(embs)
            offsets.append(offsets[-1] + len(embs))
            done.append(i)
        if done != [fr["frame"] for fr in doc["frames"]]:
            raise SystemExit(f"{clip}: video ended before the last detection frame")
        emb_info = {"model": embedder.model_id(), "seconds": round(time.perf_counter() - t0, 3),
                    "rows": offsets[-1]}
    out.mkdir(parents=True)
    shutil.copyfile(detections_path, out / "detections.json")
    if embedder is not None:
        dim = embedder.rec.embedding_dim
        np.savez(out / "embeddings.npz", emb=np.array(rows, np.float32).reshape(-1, dim),
                 offsets=np.array(offsets, np.int64))
        emb_sha = sha256_file(out / "embeddings.npz")
    manifest = {"schema": CACHE_SCHEMA[0], "schema_version": CACHE_SCHEMA[1], "clip": clip,
                "video_sha256": doc["video"]["sha256"], "detections_sha256": sha256_file(out / "detections.json"),
                "detections_provenance": doc["provenance"], "detection_config": doc["detection_config"],
                "model": doc["model"], "embeddings_sha256": emb_sha, "embedder": emb_info}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return out
