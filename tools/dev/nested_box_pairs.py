# DEV diagnostic script (not product code).
# usage: python tools/dev/nested_box_pairs.py LABEL_DIR DETECTIONS
# DEV: nested same-column box pairs vs labels. Pair (A part, B whole): IoA(A in B) >= IOA, wA/wB >= WR, hB/hA >= HR.
import json, sys, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from tools.benchmark import load_frozen
from spana.evaluation import NESTED_V1, eligible_matrix
from scipy.optimize import linear_sum_assignment
def ioa(a, b):
    iw = max(0., min(a[2], b[2]) - max(a[0], b[0])); ih = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
lab, detp = sys.argv[1:3]
fz, gt = load_frozen(Path(lab)); dets = {f['frame']: f['detections'] for f in json.load(open(detp))['frames']}
def owner(f, b):
    """label id whose box best matches b under NESTED_V1 among labelled players (None if none)"""
    P = gt.players[f]; ids = sorted(P)
    if not ids: return None
    ok, iou = eligible_matrix(np.array([P[i] for i in ids], float), np.array([b], float), NESTED_V1)
    cand = [(iou[k, 0], ids[k]) for k in range(len(ids)) if ok[k, 0]]
    return max(cand)[1] if cand else None
def centre_people(f, b):
    return [i for i, g in gt.players[f].items() if b[0] <= (g[0]+g[2])/2 <= b[2] and b[1] <= (g[1]+g[3])/2 <= b[3]]
for IOA, WR, HR in ((0.85, 0.8, 1.2), (0.85, 0.7, 1.2), (0.9, 0.8, 1.3)):
    stats = {"pairs": 0, "same_person": 0, "part_is_other_person": 0, "unlabelled": 0, "part_score_higher": 0}
    for f in gt.frames:
        D = dets[f]
        for a, A in enumerate(D):
            for b, B in enumerate(D):
                if a == b: continue
                wa, ha, wb, hb = A[2]-A[0], A[3]-A[1], B[2]-B[0], B[3]-B[1]
                if ioa(A[:4], B[:4]) >= IOA and wa / wb >= WR and hb / ha >= HR:
                    stats["pairs"] += 1; stats["part_score_higher"] += A[4] > B[4]
                    oa, ob = owner(f, A[:4]), owner(f, B[:4])
                    ppl = centre_people(f, B[:4])
                    if not ppl: stats["unlabelled"] += 1
                    elif len(ppl) == 1: stats["same_person"] += 1
                    else:
                        # two labelled people centred inside the whole box: is the part the other one?
                        stats["part_is_other_person"] += oa is not None and oa != ob
    print(f"ioa>={IOA} w_ratio>={WR} h_ratio>={HR}: {stats}")
