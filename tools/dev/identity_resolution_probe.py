# DEV diagnostic script (not product code). Replays tracking exactly as `spana track` (hash-checked against a
# reference tracks artifact) with a read-only probe on IDENTITY_UNCERTAIN resolution. For every frame on which a
# track is uncertain it records why resolution evidence did / did not advance, then classifies each track:
#   RESOLVED | RIVAL_NO_APPEARANCE | TRACK_ENDS_FIRST | NO_CLEAN_CROP | MARGIN_TOO_SMALL | OWN_SIMILARITY_LOW | OTHER
#   python -m tools.dev.identity_resolution_probe VIDEO CONFIG DETECTIONS REF_TRACKS OUT_JSON
import collections
import json
import sys
from pathlib import Path

import numpy as np

from spana.appearance import similarity
from spana.artifacts import sha256_doc
from spana.tracking import tracker as T
from tools.dev import motion_gate_probe as M

ROWS = collections.defaultdict(list)      # track_id -> per-frame rows while uncertain
_orig_res = T.MultiPlayerTracker._identity_resolution


def _probe(self, matched, dets, desc, time_s, out):
    by_track = {self._tracks[k].track_id: (k, i, c) for k, (i, c) in matched.items()}
    for t in self._tracks:
        if not t.identity_rivals:
            continue
        row = {"frame": self._frame, "t": round(time_s, 4), "state": t.state.value,
               "rivals": sorted(t.identity_rivals),
               "rival_without_appearance": sorted(r for r, a in t.identity_rivals.items() if a is None),
               "own_appearance": t.appearance is not None}
        if t.track_id not in by_track:
            row["reason"] = "UNOBSERVED"
        else:
            k, i, contested = by_track[t.track_id]
            if contested:
                row["reason"] = "CONTESTED_MATCH"
            elif desc[i] is None:
                row["reason"] = "NO_CLEAN_CROP"          # crop overlaps another detection (or too small)
            elif not self._clean(i, dets, matched):
                row["reason"] = "BOX_OVERLAPS_MATCHED"
            else:
                s_own = similarity(desc[i], t.appearance)
                riv = {r: (None if a is None else similarity(desc[i], a)) for r, a in t.identity_rivals.items()}
                row.update({"own_similarity": s_own, "rival_similarity": riv})
                if s_own is None:
                    row["reason"] = "OWN_APPEARANCE_UNKNOWN"
                elif any(v is None for v in riv.values()):
                    row["reason"] = "RIVAL_NO_APPEARANCE"
                elif s_own < self.ac.min_similarity:
                    row["reason"] = "OWN_SIMILARITY_LOW"
                elif any(s_own - v < self.ac.min_margin for v in riv.values()):
                    row["reason"] = "MARGIN_TOO_SMALL"
                    row["margin"] = round(min(s_own - v for v in riv.values()), 4)
                else:
                    row["reason"] = "FAVOURS_OWN"
                    row["margin"] = round(min(s_own - v for v in riv.values()), 4)
        ROWS[t.track_id].append(row)
    return _orig_res(self, matched, dets, desc, time_s, out)


def classify(rows, resolved, resolve_s):
    reasons = collections.Counter(r["reason"] for r in rows)
    if resolved:
        return "RESOLVED"
    if any(r["rival_without_appearance"] for r in rows) and not reasons.get("FAVOURS_OWN") and \
            reasons.get("RIVAL_NO_APPEARANCE", 0) + reasons.get("NO_CLEAN_CROP", 0) >= 0:
        if all(r["rival_without_appearance"] for r in rows):
            return "RIVAL_NO_APPEARANCE"
    observed = [r for r in rows if r["reason"] != "UNOBSERVED"]
    span = rows[-1]["t"] - rows[0]["t"] if rows else 0.0
    if span < resolve_s or not observed:
        return "TRACK_ENDS_FIRST"
    evaluable = [r for r in observed if "own_similarity" in r]
    if not evaluable:
        return "NO_CLEAN_CROP"
    top = collections.Counter(r["reason"] for r in evaluable).most_common(1)[0][0]
    return {"MARGIN_TOO_SMALL": "MARGIN_TOO_SMALL", "OWN_SIMILARITY_LOW": "OWN_SIMILARITY_LOW",
            "RIVAL_NO_APPEARANCE": "RIVAL_NO_APPEARANCE", "FAVOURS_OWN": "OTHER"}.get(top, "OTHER")


def main(video, cfg_path, det_path, ref_tracks, out_json):
    T.MultiPlayerTracker._identity_resolution = _probe
    cfg, ddoc, frames, log = M.replay(video, cfg_path, det_path)
    if sha256_doc(frames) != sha256_doc(json.load(open(ref_tracks))["frames"]):
        raise SystemExit("probe changed behaviour")
    resolved = {d["track_id"] for d in log if d["kind"] == "identity_resolved"}
    ended = {}
    for fr in frames:
        for tid, reason in fr["ended"]:
            ended[tid] = (fr["frame"], reason)
    tracks = []
    for tid, rows in ROWS.items():
        reasons = collections.Counter(r["reason"] for r in rows)
        margins = [r["margin"] for r in rows if "margin" in r]
        tracks.append({"track_id": tid, "class": classify(rows, tid in resolved, cfg.tracking.identity_resolve_s),
                       "uncertain_frames": len(rows), "uncertain_s": round(rows[-1]["t"] - rows[0]["t"], 3),
                       "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
                       "ended": ended.get(tid), "reasons": dict(reasons),
                       "rivals": sorted({x for r in rows for x in r["rivals"]}),
                       "rival_without_appearance": sorted({x for r in rows for x in r["rival_without_appearance"]}),
                       "own_appearance_known": rows[0]["own_appearance"],
                       "margin_median": None if not margins else round(float(np.median(margins)), 4),
                       "margin_max": None if not margins else round(max(margins), 4)})
    total = collections.Counter(t["class"] for t in tracks)
    frames_by_reason = collections.Counter(r["reason"] for rows in ROWS.values() for r in rows)
    Path(out_json).write_text(json.dumps({"by_class": dict(total), "frames_by_reason": dict(frames_by_reason),
                                          "tracks": tracks, "rows": ROWS}, indent=1, default=str))
    print(json.dumps({"tracks": len(tracks), "by_class": dict(total), "frames_by_reason": dict(frames_by_reason)},
                     indent=1))
    for t in sorted(tracks, key=lambda x: x["first_frame"]):
        print(f"  {t['track_id']} {t['class']:20s} {t['first_frame']}-{t['last_frame']} ({t['uncertain_s']}s) "
              f"ended={t['ended']} rivals={t['rivals']} no_app={t['rival_without_appearance']} "
              f"own_app={t['own_appearance_known']} margin med/max={t['margin_median']}/{t['margin_max']} {t['reasons']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
