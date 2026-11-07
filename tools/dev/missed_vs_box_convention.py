# DEV diagnostic script (not product code). Reads /tmp/dev/runs/F02.* and the frozen FRESH_02 labels.
import json, numpy as np, collections
from spana.labels import parse_mot
from spana.geometry import iou_matrix
L='data/labels/FRESH_02_G1_3000_midfield_buildup_passing/v1'
rec=json.load(open(L+'/labels_frozen.json'))
gt=parse_mot(open(L+'/gt.txt').read(),1920,1080,rec['frames_labelled'])
dd=json.load(open('/tmp/dev/runs/F02.det.json')); dets={fr['frame']:np.array([r[:5] for r in fr['detections']]).reshape(-1,5) for fr in dd['frames']}
r=json.load(open('/tmp/dev/runs/F02.benchmark.json'))
miss=r['events']['missed']; cat=collections.Counter(); ratios=[]; ex=[]
for m in miss:
    g=np.array(gt.players[m['frame']][m['player']]); D=dets[m['frame']]
    if not len(D): cat['no_detections_in_frame']+=1; continue
    ious=iou_matrix(g[None],D[:,:4])[0]; k=int(ious.argmax())
    gh=g[3]-g[1]; gw=g[2]-g[0]; dh=D[k,3]-D[k,1]; dw=D[k,2]-D[k,0]
    gc=(g[:2]+g[2:4])/2; dc=(D[k,:2]+D[k,2:4])/2; dist=np.hypot(*(gc-dc))/gh
    key='det_iou_0.3-0.5' if ious[k]>=0.3 else ('det_iou_0.1-0.3' if ious[k]>=0.1 else 'no_overlapping_det')
    cat[key]+=1
    if ious[k]>=0.1: ratios.append((dh/gh, dw/gw, dist, D[k,4]))
    ex.append((m['frame'],m['player'],round(float(ious[k]),2),[round(x) for x in g],[round(x) for x in D[k,:4]],round(float(D[k,4]),2)))
print('missed causes', dict(cat))
a=np.array(ratios)
if len(a): print('nearest det vs GT (iou>=0.1): height ratio med %.2f width ratio med %.2f centre dist(h) med %.2f score med %.2f'%tuple(np.median(a,axis=0)))
hr=[];wr=[];tops=[];bots=[]
for f,pl in gt.players.items():
    D=dets[f]
    if not len(D): continue
    for pid,b in pl.items():
        b=np.array(b); ious=iou_matrix(b[None],D[:,:4])[0]; k=int(ious.argmax())
        if ious[k]>=0.5:
            h=b[3]-b[1]; hr.append((D[k,3]-D[k,1])/h); wr.append((D[k,2]-D[k,0])/(b[2]-b[0])); tops.append((D[k,1]-b[1])/h); bots.append((D[k,3]-b[3])/h)
print('matched n=%d: det/gt height med %.2f width med %.2f top offset %.2f h bottom offset %.2f h'%(len(hr),np.median(hr),np.median(wr),np.median(tops),np.median(bots)))
gh=np.array([b[3]-b[1] for pl in gt.players.values() for b in pl.values()]); gw=np.array([b[2]-b[0] for pl in gt.players.values() for b in pl.values()])
print('GT box h med %d w med %d aspect %.2f'%(np.median(gh),np.median(gw),np.median(gh/gw)))
dh=np.array([q[3]-q[1] for fr in dd['frames'] for q in fr['detections'] if q[4]>=0.5]); dw=np.array([q[2]-q[0] for fr in dd['frames'] for q in fr['detections'] if q[4]>=0.5])
print('dets(>=0.5) h med %d w med %d aspect %.2f'%(np.median(dh),np.median(dw),np.median(dh/dw)))
json.dump(ex,open('/tmp/dev/f02_missed_examples.json','w'))
print('per-frame missed counts (first 20 labelled frames):',[sum(1 for m in miss if m['frame']==f) for f in sorted(gt.players)[:20]])
