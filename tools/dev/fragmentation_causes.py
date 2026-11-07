# DEV diagnostic script (not product code). For each ground-truth player whose matched track changes between two
# usage: python tools/dev/fragmentation_causes.py LABEL_DIR TRACKS.json DECISIONS.jsonl
# labelled frames, find the first frame the old track lost its detection and report the tracker's logged reason.
import collections, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '.')
from tools.benchmark import load_frozen
from spana.evaluation import NESTED_V1, eligible_matrix

labels, trk_path, dec_path = sys.argv[1:4]
fz, gt = load_frozen(Path(labels))
td = json.load(open(trk_path)); trk = {fr['frame']: fr for fr in td['frames']}
unm = collections.defaultdict(dict); other = collections.defaultdict(list)
for l in open(dec_path):
    d = json.loads(l)
    if d['kind'] == 'unmatched': unm[d['frame']][d['track_id']] = d
    else: other[d['frame']].append(d)
assign = collections.defaultdict(list)
for f in gt.frames:
    gids = sorted(gt.players[f]); obs = [t for t in trk[f]['tracks'] if t['source'] == 'OBSERVED']
    if not gids or not obs: continue
    ok, iou = eligible_matrix(np.array([gt.players[f][g] for g in gids], float), np.array([t['box'] for t in obs], float), NESTED_V1)
    r, c = linear_sum_assignment(-np.where(ok, iou + 1, 0))
    for a, b in zip(r, c):
        if ok[a, b]: assign[gids[a]].append((f, obs[b]['track_id']))
reasons = collections.Counter(); detail = collections.Counter(); rows = []
for g, seq in assign.items():
    for (f0, t0), (f1, t1) in zip(seq, seq[1:]):
        if t0 == t1: continue
        first = next(((f, unm[f][t0]) for f in range(f0 + 1, f1 + 1) if t0 in unm[f]), None)
        if first is None:
            ended = [(f, d['reason']) for f in range(f0 + 1, f1 + 1) for d in other[f] if d['kind'] == 'end' and d.get('track_id') == t0]
            key = 'ENDED_WHILE_OBSERVED:' + (ended[0][1] if ended else 'none')
            reasons[key] += 1; rows.append((g, f0, t0, f1, t1, key)); continue
        f, u = first
        key = u['reason'].split(':')[0] + (':' + u['reason'].split(':', 1)[1] if ':' in u['reason'] else '')
        reasons[key] += 1
        c0 = u['candidates'][0] if u['candidates'] else None
        if c0: detail[(key, 'nearest dist_h<=0.35' if c0['dist_h'] <= 0.35 else 'nearest dist_h>0.35', 'lowscore' if c0['low_score'] else 'highscore')] += 1
        rows.append((g, f0, t0, f, u['stage'], u['reason'], None if not c0 else {k: c0[k] for k in ('score', 'dist_h', 'gate_radius_h', 'mahalanobis2', 'height_ratio', 'iou', 'appearance_similarity', 'taken_by', 'failed')}))
total = sum(reasons.values())
print(f'track changes: {total}')
for k, v in reasons.most_common(): print(f'{v:4d} {100*v/total:5.1f}%  {k}')
print('detail:'); [print(f'{v:4d} {k}') for k, v in detail.most_common(8)]
json.dump(rows, open('/tmp/dev/f02_frag_rows.json', 'w'), default=str)
