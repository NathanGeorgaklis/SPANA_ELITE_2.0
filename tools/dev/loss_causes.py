# DEV diagnostic script (not product code). Reads runs from /tmp/dev/runs and clips from the staged DEV folder.
import json, sys, numpy as np, collections
from spana.geometry import iou_matrix
n=sys.argv[1]; t=json.load(open(f'/tmp/dev/runs/{n}.trk.json'))
prev=None; c=collections.Counter(); first={}
for fr in t['frames']:
    for x in fr['tracks']: first.setdefault(x['track_id'],fr['frame'])
for fr in t['frames']:
    cur={x['track_id']:x for x in fr['tracks']}
    if prev:
        for tid,p in prev.items():
            if p['source']=='OBSERVED' and p['confirmed'] and tid in cur and cur[tid]['source']!='OBSERVED':
                pb=np.array(cur[tid]['box']); h=pb[3]-pb[1]; pc=(pb[:2]+pb[2:])/2
                obs=[x for x in cur.values() if x['source']=='OBSERVED' and x['track_id']!=tid]
                if not obs: c['nothing_near']+=1; continue
                ob=np.array([x['box'] for x in obs]); oc=(ob[:,:2]+ob[:,2:])/2
                d=np.hypot(*(oc-pc).T)/h; k=int(np.argmin(d))
                if d[k]>0.35: c['nothing_near']+=1; continue
                tk=obs[k]['track_id']
                if tk in prev and prev[tk]['source']=='OBSERVED':
                    i=iou_matrix(np.array([p['box']]),np.array([prev[tk]['box']]))[0,0]
                    c['taken_by_duplicate(iou>0.5)' if i>0.5 else 'taken_by_neighbour']+=1
                else: c['taken_by_new_track']+=1
    prev=cur
print(' losses:',dict(c),'total',sum(c.values()))
