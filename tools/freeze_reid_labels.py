"""Freeze an exported re-ID labelling session into the repository.

    python -m tools.freeze_reid_labels --pack PACK_DIR --export MATCH_reid_vN.json [--video MATCH.mp4]

Checks: the match manifest is complete (split, rights, consent; TRAIN needs training_permitted); the export belongs
to this pack (match_id, split, tracklets sha256); every proposed tracklet is fully decided by contiguous parts that
start and end on its observed frames; every player decision names a declared player with a valid side; one player
never occupies two parts on the same frame; a version > 1 carries a reason. Writes
data/reid/<split>/<match_id>/v<N>/{manifest.json, players.json, identities.jsonl, boxes.mot.txt, export.json,
FROZEN.json} with the sha256 of every file, and never overwrites an existing version.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from spana.artifacts import write_json_atomic
from spana.reid_labels import (IDENTITIES_SCHEMA, PLAYER_RE, SIDES, ReidLabelError, sha256_bytes,
                               validate_manifest)
from spana.video import sha256_file

REPO = Path(__file__).resolve().parent.parent
EXPORT_SCHEMA = ("spana.reid_label_export", "1.0.0")
DECISIONS = ("player", "reject", "unsure")


def _check_export(pack: dict, tl: Dict[str, dict], ex: dict) -> List[dict]:
    if (ex.get("schema"), ex.get("schema_version")) != EXPORT_SCHEMA:
        raise ReidLabelError("not a re-ID label export (schema)")
    m = pack["match"]
    if ex.get("match_id") != m["match_id"] or ex.get("split") != m["split"]:
        raise ReidLabelError("export is for a different match or split than the pack")
    if ex.get("tracklets_sha256") != pack["tracklets_sha256"]:
        raise ReidLabelError("export was made from a different set of tracklet proposals")
    if not str(ex.get("annotator", "")).strip():
        raise ReidLabelError("annotator missing")
    v = ex.get("label_version")
    if not (isinstance(v, int) and v >= 1):
        raise ReidLabelError("label_version must be an integer >= 1")
    if v > 1 and not str(ex.get("reason", "")).strip():
        raise ReidLabelError("a new label version needs a reason")
    players = ex.get("players") or {}
    for key, p in players.items():
        if not PLAYER_RE.match(key) or p.get("side") not in SIDES:
            raise ReidLabelError(f"bad player {key!r}: key [A-Za-z0-9_-]{{1,16}}, side one of {SIDES}")
    parts_by_tl: Dict[str, List[dict]] = {}
    for part in ex.get("parts", []):
        parts_by_tl.setdefault(part.get("tracklet"), []).append(part)
    unknown = set(parts_by_tl) - set(tl)
    if unknown:
        raise ReidLabelError(f"export names unknown tracklets: {sorted(unknown)[:5]}")
    rows: List[dict] = []
    for tid, t in tl.items():
        parts = sorted(parts_by_tl.get(tid, []), key=lambda p: p.get("first_frame", -1))
        if not parts:
            raise ReidLabelError(f"tracklet {tid} has no decision (every tracklet must be decided)")
        frames = t["frames"]
        pos = {f: k for k, f in enumerate(frames)}
        expect = 0
        for n, p in enumerate(parts):
            ff, lf = p.get("first_frame"), p.get("last_frame")
            if ff not in pos or lf not in pos or pos[ff] != expect or pos[lf] < pos[ff]:
                raise ReidLabelError(f"tracklet {tid}: parts must be contiguous and on observed frames")
            expect = pos[lf] + 1
            d = p.get("decision")
            if d not in DECISIONS:
                raise ReidLabelError(f"tracklet {tid}: undecided or unknown decision {d!r}")
            if d == "player" and p.get("player") not in players:
                raise ReidLabelError(f"tracklet {tid}: player {p.get('player')!r} is not declared")
            if d != "player" and p.get("player") is not None:
                raise ReidLabelError(f"tracklet {tid}: a {d} part cannot name a player")
            rows.append({"tracklet": tid, "part": n + 1, "decision": d, "player": p.get("player"),
                         "side": players[p["player"]]["side"] if d == "player" else None,
                         "first_frame": ff, "last_frame": lf, "observed_frames": pos[lf] - pos[ff] + 1,
                         "source_track": t["track"]})
        if expect != len(frames):
            raise ReidLabelError(f"tracklet {tid}: parts do not cover the whole tracklet")
    seen: Dict[tuple, str] = {}
    for r in rows:
        if r["decision"] != "player":
            continue
        t = tl[r["tracklet"]]
        for f in t["frames"][t["frames"].index(r["first_frame"]):t["frames"].index(r["last_frame"]) + 1]:
            k = (r["player"], f)
            if k in seen:
                raise ReidLabelError(f"player {r['player']} is in two places on frame {f} "
                                     f"({seen[k]} and {r['tracklet']}#{r['part']})")
            seen[k] = f"{r['tracklet']}#{r['part']}"
    return rows


def freeze(pack_dir: str, export_path: str, video: str | None = None, root: Path = REPO) -> Path:
    pd = Path(pack_dir)
    pack = json.loads((pd / "pack.json").read_text())
    manifest = validate_manifest(pack["match"])
    tl_bytes = (pd / "tracklets.json").read_bytes()
    if sha256_bytes(tl_bytes) != pack["tracklets_sha256"]:
        raise ReidLabelError("tracklets.json changed after the pack was made")
    if video is not None and sha256_file(video) != manifest["video_sha256"]:
        raise ReidLabelError("video does not match the match manifest")
    tl = {t["id"]: t for t in json.loads(tl_bytes)}
    ex_bytes = Path(export_path).read_bytes()
    ex = json.loads(ex_bytes)
    rows = _check_export(pack, tl, ex)
    dest = root / "data" / "reid" / manifest["split"] / manifest["match_id"] / f"v{ex['label_version']}"
    if dest.exists():
        raise ReidLabelError(f"{dest} already exists: frozen labels are never overwritten")
    used = sorted({r["player"] for r in rows if r["decision"] == "player"})
    players = {k: {"id": n + 1, "side": ex["players"][k]["side"],
                   "shirt_number": ex["players"][k].get("shirt_number")} for n, k in enumerate(used)}
    mot: List[str] = []
    for r in rows:
        if r["decision"] != "player":
            continue
        t = tl[r["tracklet"]]
        a, b = t["frames"].index(r["first_frame"]), t["frames"].index(r["last_frame"])
        for f, (x1, y1, x2, y2) in zip(t["frames"][a:b + 1], t["boxes"][a:b + 1]):
            mot.append(f"{f + 1},{players[r['player']]['id']},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,1,1")
    mot.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))
    files = {
        "manifest.json": json.dumps(manifest, indent=1, sort_keys=True).encode(),
        "players.json": json.dumps(players, indent=1, sort_keys=True).encode(),
        "identities.jsonl": "".join(json.dumps({"schema": IDENTITIES_SCHEMA[0], **r}, sort_keys=True) + "\n"
                                    for r in rows).encode(),
        "boxes.mot.txt": ("\n".join(mot) + "\n").encode() if mot else b"",
        "export.json": ex_bytes,
    }
    dest.mkdir(parents=True)
    for name, data in files.items():
        (dest / name).write_bytes(data)
    count = lambda d: sum(1 for r in rows if r["decision"] == d)  # noqa: E731
    write_json_atomic(dest / "FROZEN.json", {
        "schema": "spana.reid_frozen", "schema_version": "1.0.0", "split": manifest["split"],
        "match_id": manifest["match_id"], "label_version": ex["label_version"], "reason": ex.get("reason", ""),
        "annotator": ex["annotator"], "exported_utc": ex.get("exported_utc"),
        "frozen_in_repo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_sha256": manifest["video_sha256"], "tracks_sha256": pack["tracks_sha256"],
        "tracklets_sha256": pack["tracklets_sha256"], "proposal": pack["proposal"],
        "counts": {"tracklets": len(tl), "parts": len(rows), "player_parts": count("player"),
                   "rejected_parts": count("reject"), "unsure_parts": count("unsure"), "players": len(players),
                   "boxes": len(mot)},
        "files": {name: sha256_bytes(data) for name, data in files.items()},
        "rule": "Frozen. Never edit; a correction is a new version with a reason. EVAL is never used for "
                "training or tuning (spana.reid_labels.split_guard).",
    })
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--export", required=True)
    p.add_argument("--video", default=None)
    a = p.parse_args()
    try:
        dest = freeze(a.pack, a.export, a.video)
    except ReidLabelError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"frozen -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
