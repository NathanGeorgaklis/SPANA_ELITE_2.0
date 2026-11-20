"""Speed profile summary and identity comparison for a speed change (DEV only; measurement, not product code).

    python -m tools.identity_speed profile --config C --profile det.profile.json --profile trk.profile.json \
        [--profile ...] --out SUMMARY.json
    python -m tools.identity_speed baseline --config C --tracks BASE.trk.json --out BASE.identity.json
    python -m tools.identity_speed compare --config C --baseline BASE.identity.json --base-tracks BASE.trk.json \
        --tracks NEW.trk.json --out COMPARE.json

profile   per-pass and per-stage ms per processed frame and share, with one-off costs (model load, hashing the
          video, decoding up to a window's start) kept apart from the per-frame cost; the per-frame cost projected to
          `performance.projection_footage_s` and compared with `performance.realtime_factor_target`.
baseline  identity outputs to compare against. Probes = simulated taps: at frames `performance.identity_probe_every_s`
          apart, every track that is ACTIVE, confirmed and CERTAIN (a box the app would let the user tap). Each probe
          is followed with spana.identity.follower (default config, no user actions) to the end of the clip. Saved per
          probe and frame: state, reason, answered track, box, and the answered track's tracker confidence (identity
          confidence); plus the LOST / ABSTAIN / CANDIDATE gaps and each reacquisition (gap ending in TARGET).
compare   the same probes on the new tracks. A probe's tap is reproduced on the new tracks as the unique tappable
          track whose box overlaps the baseline tap box at IoU >= `performance.identity_box_iou_min`; otherwise the
          probe is "not reproducible" and all its frames disagree. Per frame: agree = same state and, when both
          show a box, IoU >= identity_box_iou_min. `conflicting answers` = both TARGET on boxes with IoU < 0.5 (the
          answer moved to another player). Pass/fail against config `performance`:
            agreement >= identity_agreement_min; conflicting answers == 0 ("identity switches must not increase");
            LOST + ABSTAIN share rises by at most lost_unknown_rate_max_increase.
Labelled identity errors (wrong player-frames, tracker ID switches) come from tools/target_score.py and
tools/benchmark.py on the labelled DEV clips; this tool needs no labels.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from typing import Dict, List, Optional

import numpy as np

from spana.artifacts import read_json, sha256_doc, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.geometry import iou_matrix
from spana.identity.follower import Selection, follow

ONE_OFF = ("setup.model_load", "io.video_sha256", "decode.seek_to_start", "io.read_detections", "io.read_tracks")
NO_ANSWER = ("LOST", "ABSTAIN")
CONFLICT_IOU = 0.5   # reporting convention: two boxes under this IoU are different players


def _iou(a, b) -> float:
    return float(iou_matrix(np.array([a], float), np.array([b], float))[0, 0])


# ---------------------------------------------------------------------------------------------------- profile
def summarize_profiles(profiles: List[dict], cfg) -> dict:
    per_pass, stages, total_ms = [], [], 0.0
    one_off_s = 0.0
    for p in profiles:
        n = max(1, p["frames"])
        rec = [s for s in p["stages"] if s["stage"] not in ONE_OFF]
        oo = sum(s["seconds"] for s in p["stages"] if s["stage"] in ONE_OFF)
        untimed = max(0.0, p["wall_s"] - p["timed_s"])
        ms = 1000.0 * (sum(s["seconds"] for s in rec) + untimed) / n
        one_off_s += oo
        total_ms += ms
        per_pass.append({"pass": p["pass"], "frames": p["frames"], "ms_per_frame": round(ms, 3),
                         "one_off_s": round(oo, 3), "untimed_ms_per_frame": round(1000 * untimed / n, 3),
                         "info": p.get("info", {})})
        stages += [{"pass": p["pass"], "stage": s["stage"], "ms_per_frame": round(1000 * s["seconds"] / n, 3)}
                   for s in rec]
    for s in stages:
        s["share"] = round(s["ms_per_frame"] / total_ms, 4) if total_ms else 0.0
    stages.sort(key=lambda s: -s["ms_per_frame"])
    fps = next((p["info"]["video"]["fps"] for p in profiles if "video" in p.get("info", {})), None)
    pc = cfg.performance
    out = {"ms_per_frame": round(total_ms, 3), "stages": stages, "passes": per_pass,
           "one_off_s_on_this_clip": round(one_off_s, 3), "fps": fps,
           "projection_footage_s": pc.projection_footage_s,
           "realtime_factor_target": pc.realtime_factor_target}
    if fps:
        rt = total_ms / 1000.0 * fps
        out["realtime_factor"] = round(rt, 3)
        out["projected_hours"] = round(rt * pc.projection_footage_s / 3600.0, 3)
        out["meets_target"] = rt <= pc.realtime_factor_target
        out["ms_per_frame_budget"] = round(1000.0 * pc.realtime_factor_target / fps, 3)
    return out


# --------------------------------------------------------------------------------------------------- identity
def probes(tracks_doc: dict, every_s: float) -> List[dict]:
    out, next_t = [], None
    for fr in tracks_doc["frames"]:
        if next_t is not None and fr["t"] < next_t - 1e-9:
            continue
        taps = [t for t in fr["tracks"] if t["state"] == "ACTIVE" and t["confirmed"] and t["identity"] == "CERTAIN"]
        if not taps:
            continue
        next_t = fr["t"] + every_s
        out += [{"frame": fr["frame"], "t": fr["t"], "track_id": t["track_id"], "box": t["box"]} for t in taps]
    return out


def _gaps(frames: List[dict]) -> List[dict]:
    gaps, cur = [], None
    for fr in frames:
        if fr["state"] != "TARGET":
            if cur is None:
                cur = {"start_frame": fr["frame"], "start_t": fr["t"], "states": set()}
            cur["states"].add(fr["state"])
            cur["end_frame"], cur["end_t"] = fr["frame"], fr["t"]
        elif cur is not None:
            cur["reacquired_t"] = fr["t"]
            gaps.append(cur)
            cur = None
    if cur is not None:
        cur["reacquired_t"] = None
        gaps.append(cur)
    for g in gaps:
        g["states"] = sorted(g["states"])
        g["latency_s"] = None if g["reacquired_t"] is None else round(g["reacquired_t"] - g["start_t"], 6)
    return gaps


_CONF: Dict[int, dict] = {}


def _confidences(tracks_doc: dict) -> dict:
    key = id(tracks_doc)
    if key not in _CONF:
        _CONF[key] = {(fr["frame"], t["track_id"]): t["confidence"] for fr in tracks_doc["frames"]
                      for t in fr["tracks"]}
    return _CONF[key]


def follow_probe(tracks_doc: dict, probe: dict, track_id: str, cfg) -> dict:
    frames, info = follow(tracks_doc, Selection.track_at_frame(track_id, probe["frame"]), cfg, ())
    conf = _confidences(tracks_doc)
    rows = [{"frame": f["frame"], "t": f["t"], "state": f["state"], "reason": f["reason"], "track_id": f["track_id"],
             "box": f["box"], "identity_confidence": conf.get((f["frame"], f["track_id"]))} for f in frames]
    return {"frames": rows, "gaps": _gaps(rows), "relinks": len(info["relinks"]), "track_ends": info["track_ends"]}


def identity_doc(tracks_doc: dict, cfg) -> dict:
    ps = probes(tracks_doc, cfg.performance.identity_probe_every_s)
    return {"schema": "spana.identity_speed_baseline", "schema_version": "1.0.0",
            "tracks_sha256": sha256_doc(tracks_doc), "probe_every_s": cfg.performance.identity_probe_every_s,
            "probes": [{**p, **follow_probe(tracks_doc, p, p["track_id"], cfg)} for p in ps],
            "summary": _summary(tracks_doc)}


def _summary(tracks_doc: dict) -> dict:
    ids, n_tf, n_unc = set(), 0, 0
    for fr in tracks_doc["frames"]:
        for t in fr["tracks"]:
            ids.add(t["track_id"])
            if t["state"] == "ACTIVE" and t["confirmed"]:
                n_tf += 1
                n_unc += t["identity"] != "CERTAIN"
    return {"frames": len(tracks_doc["frames"]), "tracks": len(ids), "tracklets": len(tracks_doc["tracklets"]),
            "confirmed_observed_track_frames": n_tf, "identity_uncertain_share": round(n_unc / max(1, n_tf), 6)}


def _rates(probe_runs: List[dict]) -> dict:
    n = sum(len(p["frames"]) for p in probe_runs)
    cnt = {s: sum(1 for p in probe_runs for f in p["frames"] if f["state"] == s)
           for s in ("TARGET", "CANDIDATE", "ABSTAIN", "LOST")}
    lat = [g["latency_s"] for p in probe_runs for g in p["gaps"] if g["latency_s"] is not None]
    return {"frames": n, **{f"{k.lower()}_share": round(v / max(1, n), 6) for k, v in cnt.items()},
            "lost_unknown_share": round((cnt["LOST"] + cnt["ABSTAIN"]) / max(1, n), 6),
            "gaps": sum(len(p["gaps"]) for p in probe_runs), "reacquisitions": len(lat),
            "reacquisition_latency_s_median": round(statistics.median(lat), 4) if lat else None,
            "reacquisition_latency_s_mean": round(statistics.mean(lat), 4) if lat else None}


def compare(base: dict, base_tracks: dict, new_tracks: dict, cfg) -> dict:
    pc = cfg.performance
    if base["tracks_sha256"] != sha256_doc(base_tracks):
        raise SystemExit("baseline identity file was not made from --base-tracks")
    by_frame = {fr["frame"]: fr for fr in new_tracks["frames"]}
    if [fr["frame"] for fr in base_tracks["frames"]] != list(by_frame):
        raise SystemExit("tracks artifacts cover different frames")
    agree = total = conflicts = state_diff = box_diff = 0
    not_repro, new_runs, per_probe = [], [], []
    for p in base["probes"]:
        taps = [t for t in by_frame[p["frame"]]["tracks"] if t["state"] == "ACTIVE" and t["confirmed"]
                and t["identity"] == "CERTAIN" and _iou(t["box"], p["box"]) >= pc.identity_box_iou_min]
        n = len(p["frames"])
        total += n
        if len(taps) != 1:
            not_repro.append({"frame": p["frame"], "track_id": p["track_id"], "matches": len(taps)})
            per_probe.append({"frame": p["frame"], "track_id": p["track_id"], "agreement": 0.0})
            continue
        run = follow_probe(new_tracks, p, taps[0]["track_id"], cfg)
        new_runs.append(run)
        a = c = 0
        for fb, fn in zip(p["frames"], run["frames"]):
            if fb["state"] != fn["state"]:
                state_diff += 1
                continue
            if fb["box"] is None or _iou(fb["box"], fn["box"]) >= pc.identity_box_iou_min:
                a += 1
                continue
            box_diff += 1
            if fb["state"] == "TARGET" and _iou(fb["box"], fn["box"]) < CONFLICT_IOU:
                c += 1
        agree += a
        conflicts += c
        per_probe.append({"frame": p["frame"], "track_id": p["track_id"], "new_track_id": taps[0]["track_id"],
                          "agreement": round(a / max(1, n), 6), "conflicting_answers": c})
    b_rates = _rates(base["probes"])
    reproduced = {(p["frame"], p["track_id"]) for p in per_probe if "new_track_id" in p}
    b_rates_repro = _rates([p for p in base["probes"] if (p["frame"], p["track_id"]) in reproduced])
    n_rates = _rates(new_runs)
    agreement = agree / max(1, total)
    rise = n_rates["lost_unknown_share"] - b_rates_repro["lost_unknown_share"]
    checks = {"agreement": {"value": round(agreement, 6), "min": pc.identity_agreement_min,
                            "ok": agreement >= pc.identity_agreement_min},
              "conflicting_answers": {"value": conflicts, "max": 0, "ok": conflicts == 0},
              "lost_unknown_rise": {"value": round(rise, 6), "max": pc.lost_unknown_rate_max_increase,
                                    "ok": rise <= pc.lost_unknown_rate_max_increase}}
    return {"schema": "spana.identity_speed_compare", "schema_version": "1.0.0",
            "base_tracks_sha256": base["tracks_sha256"], "new_tracks_sha256": sha256_doc(new_tracks),
            "identical_tracks": base["tracks_sha256"] == sha256_doc(new_tracks),
            "probes": len(base["probes"]), "probes_not_reproducible": not_repro, "frames_compared": total,
            "agree_frames": agree, "state_differs": state_diff, "box_differs": box_diff,
            "conflicting_answers": conflicts, "baseline": b_rates, "baseline_reproduced_probes": b_rates_repro,
            "new": n_rates, "tracker_baseline": base["summary"], "tracker_new": _summary(new_tracks),
            "checks": checks, "pass": all(c["ok"] for c in checks.values()), "per_probe": per_probe}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="identity_speed")
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("profile")
    a1.add_argument("--profile", action="append", required=True)
    a2 = sub.add_parser("baseline")
    a2.add_argument("--tracks", required=True)
    a3 = sub.add_parser("compare")
    a3.add_argument("--baseline", required=True)
    a3.add_argument("--base-tracks", required=True)
    a3.add_argument("--tracks", required=True)
    for s in (a1, a2, a3):
        s.add_argument("--config", required=True)
        s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "profile":
        doc = summarize_profiles([read_json(x) for x in a.profile], cfg)
    elif a.cmd == "baseline":
        t = read_json(a.tracks)
        validate_tracks(t)
        doc = identity_doc(t, cfg)
    else:
        bt, nt = read_json(a.base_tracks), read_json(a.tracks)
        validate_tracks(bt)
        validate_tracks(nt)
        doc = compare(read_json(a.baseline), bt, nt, cfg)
    doc["argv"] = sys.argv
    print(write_json_atomic(a.out, doc))
    if a.cmd == "compare":
        print({k: v for k, v in doc.items() if k in ("identical_tracks", "checks", "pass")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
