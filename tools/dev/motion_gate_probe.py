# DEV diagnostic script (not product code). Replays tracking exactly as `spana track` does, with READ-ONLY probes:
#  * every accepted match: normalised innovation squared (NIS) of the detection centre against the track's predicted
#    position covariance + measurement noise (the quantity the Mahalanobis gate tests), with dt and streak length;
#  * every logged gate report: predicted/detection centre, Mahalanobis^2, threshold, S, P, dt, observed streak.
# Then, for every labelled-player track change whose first cause is a failed Mahalanobis gate, prints those values,
# the competing candidates, and whether the rejected detection is the player's true continuation.
# The replayed tracks must hash-equal the reference tracks artifact (probes cannot change behaviour).
#   python -m tools.dev.motion_gate_probe LABEL_DIR VIDEO CONFIG DETECTIONS REF_TRACKS OUT_JSON
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from spana.appearance import describe
from spana.artifacts import detections_from_frame, read_json, sha256_doc, tracks_frame
from spana.config import load_config
from spana.evaluation import NESTED_V1, eligible_matrix
from spana.tracking import tracker as T
from spana.tracking.camera import CameraMotionEstimator
from spana.geometry import box_to_cxcywh
from spana.video import VideoSource
from tools.benchmark import load_frozen

PROBE = {"nis": [], "gates": {}}
_orig_gate, _orig_apply = T.MultiPlayerTracker._gate_report, T.MultiPlayerTracker._apply_matches


def _kf_view(t, d, time_s):
    kf = t.kf
    pc, dc = kf.x[:2].copy(), box_to_cxcywh(d.as_array())[:2]
    R = (kf.cfg.pos_std_h * kf.height) ** 2
    S = kf.P[:2, :2] + np.eye(2) * R
    return {"pred_c": [round(float(v), 1) for v in pc], "det_c": [round(float(v), 1) for v in dc],
            "innov_h": [round(float(v), 4) for v in (dc - pc) / kf.height],
            "pred_h": round(float(kf.height), 1),
            "sqrtP_pos_h": [round(float(np.sqrt(kf.P[i, i])) / kf.height, 4) for i in (0, 1)],
            "sqrtP_vel_h_per_s": [round(float(np.sqrt(kf.P[i, i])) / kf.height, 4) for i in (4, 5)],
            "sqrtR_h": round(float(np.sqrt(R)) / kf.height, 4),
            "sqrtS_h": [round(float(np.sqrt(S[i, i])) / kf.height, 4) for i in (0, 1)],
            "vel_h_per_s": [round(float(v) / kf.height, 3) for v in kf.x[4:6]],
            "dt_since_observed_s": round(time_s - t.last_observed_t, 4),
            "observed_frames": t.observed_frames, "age_s": round(time_s - t.first_observed_t, 3)}


def _gate(self, t, d, desc_i, time_s):
    r = _orig_gate(self, t, d, desc_i, time_s)
    PROBE["gates"][(self._frame, t.track_id, tuple(round(v, 1) for v in d.box))] = \
        {**_kf_view(t, d, time_s), "gate_chi2": self.tc.gate_chi2}
    return r


def _apply(self, matched, dets, desc, frame_index, time_s):
    for k, (i, _) in matched.items():
        t = self._tracks[k]
        if t.observed_last_frame and t.confirmed:            # continuous tracking only
            v = _kf_view(t, dets[i], time_s)
            m2 = t.kf.position_mahalanobis2(box_to_cxcywh(dets[i].as_array())[:2])
            PROBE["nis"].append((m2, v["dt_since_observed_s"], t.observed_frames, v["sqrtS_h"][0], v["sqrtS_h"][1],
                                 abs(v["innov_h"][0]), abs(v["innov_h"][1]), self._frame, v["innov_h"][0], v["innov_h"][1]))
    return _orig_apply(self, matched, dets, desc, frame_index, time_s)


def replay(video_path, cfg_path, det_path):
    cfg = load_config(cfg_path)
    video, ddoc = VideoSource(video_path), read_json(det_path)
    det_frames = {fr["frame"]: fr for fr in ddoc["frames"]}
    cam = CameraMotionEstimator(cfg.camera, video.meta.width, video.meta.height, video.meta.fps)
    trk = T.MultiPlayerTracker(cfg, video.meta.width, video.meta.height)
    T.MultiPlayerTracker._gate_report, T.MultiPlayerTracker._apply_matches = _gate, _apply
    frames, log = [], []
    first, last, stride = min(det_frames), max(det_frames), ddoc["frame_stride"]
    for i, t, img in video.frames(first, last + 1):
        if (i - first) % stride:
            continue
        dets = detections_from_frame(det_frames[i])
        boxes = [d.box for d in dets]
        trk._frame = i
        ft = trk.update(i, t, dets, cam.observe(img, boxes), describe(img, boxes, cfg.appearance))
        frames.append(tracks_frame(ft, cfg.output))
        log.extend({"frame": i, **d} for d in ft.decisions)
    return cfg, ddoc, frames, log


def main(label_dir, video_path, cfg_path, det_path, ref_tracks, out_json):
    cfg, ddoc, frames, log = replay(video_path, cfg_path, det_path)
    ref = read_json(ref_tracks)
    same = sha256_doc(frames) == sha256_doc(ref["frames"])
    print("replay identical to reference tracks:", same)
    if not same:
        raise SystemExit("probe changed behaviour")
    fz, gt = load_frozen(Path(label_dir))
    trk = {fr["frame"]: fr for fr in frames}
    dets = {fr["frame"]: fr["detections"] for fr in ddoc["frames"]}
    unm = collections.defaultdict(dict)
    by_frame = collections.defaultdict(list)
    for d in log:
        (unm[d["frame"]].__setitem__(d["track_id"], d) if d["kind"] == "unmatched" else by_frame[d["frame"]].append(d))
    assign = collections.defaultdict(list)
    for f in gt.frames:
        gids = sorted(gt.players[f]); obs = [t for t in trk[f]["tracks"] if t["source"] == "OBSERVED"]
        if not gids or not obs:
            continue
        ok, iou = eligible_matrix(np.array([gt.players[f][g] for g in gids], float),
                                  np.array([t["box"] for t in obs], float), NESTED_V1)
        r, c = linear_sum_assignment(-np.where(ok, iou + 1, 0))
        for a, b in zip(r, c):
            if ok[a, b]:
                assign[gids[a]].append((f, obs[b]["track_id"]))
    cases = []
    for g, seq in assign.items():
        for (f0, t0), (f1, t1) in zip(seq, seq[1:]):
            if t0 == t1:
                continue
            first = next(((f, unm[f][t0]) for f in range(f0 + 1, f1 + 1) if t0 in unm[f]), None)
            if first is None:
                continue
            f, u = first
            if not u["reason"].startswith("GATE_FAILED") or "mahalanobis_ok" not in u["reason"]:
                continue
            c0 = u["candidates"][0]
            box = tuple(round(v, 1) for v in dets[f][c0["detection"]][:4]) if "detection" in c0 else None
            # true continuation: the rejected detection is taken over by the track that later carries this player
            owner_at_f = next((o["track_id"] for o in trk[f]["tracks"] if o["source"] == "OBSERVED"
                               and o["detection_index"] == c0.get("detection")), None)
            if f in gt.players and g in gt.players[f] and box is not None:
                ok, _ = eligible_matrix(np.array([gt.players[f][g]], float), np.array([box], float), NESTED_V1)
                truth = "LABELLED_SAME_PLAYER" if ok[0, 0] else "LABELLED_OTHER"
            else:
                truth = "TAKEN_OVER_BY_SUCCESSOR" if owner_at_f == t1 else f"OWNER_AT_F={owner_at_f}"
            probe = PROBE["gates"].get((f, t0, box))
            cases.append({"player": g, "old_track": t0, "new_track": t1, "frame": f, "stage": u["stage"],
                          "reason": u["reason"], "truth": truth, "owner_at_frame": owner_at_f,
                          "gate": {k: c0[k] for k in ("score", "dist_h", "gate_radius_h", "mahalanobis2", "failed",
                                                      "taken_by", "height_ratio", "iou")},
                          "kalman": probe,
                          "competitors": [{k: c.get(k) for k in ("detection", "score", "dist_h", "mahalanobis2",
                                                                  "taken_by", "failed")} for c in u["candidates"][1:]]})
    nis = np.array(PROBE["nis"])
    summary = {"accepted_continuous_matches": len(nis),
               "nis_mean": round(float(nis[:, 0].mean()), 3), "nis_expected_mean_2dof": 2.0,
               "share_nis_over_gate": round(float((nis[:, 0] > cfg.tracking.gate_chi2).mean()), 4),
               "expected_share_over_gate": 0.01,
               "nis_p50_p90_p99": [round(float(v), 2) for v in np.percentile(nis[:, 0], [50, 90, 99])],
               "sqrtS_h_median_xy": [round(float(np.median(nis[:, 3])), 4), round(float(np.median(nis[:, 4])), 4)],
               "abs_innov_h_p50_p90_xy": [[round(float(v), 4) for v in np.percentile(nis[:, 5], [50, 90])],
                                         [round(float(v), 4) for v in np.percentile(nis[:, 6], [50, 90])]]}
    for lo, hi in ((1, 5), (5, 30), (30, 10 ** 9)):
        m = (nis[:, 2] >= lo) & (nis[:, 2] < hi)
        if m.any():
            summary[f"streak_{lo}_{hi if hi < 10**9 else 'inf'}"] = {
                "n": int(m.sum()), "nis_mean": round(float(nis[m, 0].mean()), 3),
                "sqrtS_h_median_y": round(float(np.median(nis[m, 4])), 4),
                "share_over_gate": round(float((nis[m, 0] > cfg.tracking.gate_chi2).mean()), 4)}
    for c in cases:                      # common-mode innovation of the frame's other matches (camera-model error)
        m = nis[:, 7] == c["frame"]
        c["frame_other_matches"] = {"n": int(m.sum()),
                                    "median_innov_h_xy": [round(float(np.median(nis[m, 8])), 4) if m.any() else None,
                                                          round(float(np.median(nis[m, 9])), 4) if m.any() else None]}
        cam = trk[c["frame"]]["camera"]
        c["camera"] = {"status": cam["status"], "inliers": cam["inliers"], "affine": cam["affine"]}
    # camera support vs common-mode innovation: frames binned by the camera estimate's inlier count
    per = collections.defaultdict(list)
    for row in nis:
        per[int(row[7])].append(row[8:10])
    bins = collections.defaultdict(list)
    for f, rows in per.items():
        if len(rows) >= 2:
            inl = trk[f]["camera"]["inliers"]
            b = "<50" if inl < 50 else "50-150" if inl < 150 else ">=150"
            bins[b].append(float(np.hypot(*np.median(np.array(rows), axis=0))))
    summary["common_mode_innov_h_by_inliers"] = {
        b: {"frames": len(v), "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4)}
        for b, v in sorted(bins.items())}
    Path(out_json).write_text(json.dumps({"clip": fz["clip"]["name"], "summary": summary, "cases": cases}, indent=1))
    print(json.dumps(summary, indent=1))
    for c in cases:
        k = c["kalman"] or {}
        print(f"{c['frame']:5d} {c['old_track']}->{c['new_track']} {c['truth']:24s} m2={c['gate']['mahalanobis2']:7.2f} "
              f"dist_h={c['gate']['dist_h']:.3f} innov_h={k.get('innov_h')} sqrtS_h={k.get('sqrtS_h')} "
              f"sqrtP_pos={k.get('sqrtP_pos_h')} sqrtR={k.get('sqrtR_h')} dt={k.get('dt_since_observed_s')} "
              f"streak={k.get('observed_frames')} vel={k.get('vel_h_per_s')} comp={len(c['competitors'])}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
