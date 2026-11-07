import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from spana.geometry import iou_matrix
from spana.records import PositionSource
from tests.synth import run
import adv
def diag(name, fps):
    ps = adv.S[name]()
    res = run(ps, fps=fps, duration_s=3)
    track_gt = {}
    out = []
    for f, ft in enumerate(res.frames):
        gt = res.gt_boxes[f]; gids = sorted(gt)
        obs = [o for o in ft.tracks if o.source == PositionSource.OBSERVED]
        if not gids or not obs: continue
        ious = iou_matrix(np.array([gt[g] for g in gids]), np.array([o.box for o in obs]))
        r, c = linear_sum_assignment(-ious)
        for a, b in zip(r, c):
            if ious[a, b] < 0.5: continue
            g, tid = gids[a], obs[b].track_id
            if tid in track_gt and track_gt[tid] != g:
                prev = track_gt[tid]
                dec = [d for d in ft.decisions if d.get("track_id") == tid and d["kind"] in ("match","recovery")]
                st = dec[0].get("stage") if dec else "?"
                # separation of gt players
                hb = (gt[g][3]-gt[g][1])
                sep = None
                if prev in gt:
                    cg = (gt[g][:2]+gt[g][2:])/2; cp = (gt[prev][:2]+gt[prev][2:])/2
                    sep = float(np.hypot(*(cg-cp))/hb)
                pv = prev in gt
                out.append((f, round(f/fps,3), tid, prev, g, st, dec[0].get("contested") if dec else None,
                            "prev_visible" if pv else "prev_hidden", None if sep is None else round(sep,3),
                            dec[0].get("dist_h") if dec else None, dec[0].get("gap_s") if dec else None))
            track_gt[tid] = g
    return out
if __name__ == "__main__":
    for x in diag(sys.argv[1], int(sys.argv[2])): print(x)
