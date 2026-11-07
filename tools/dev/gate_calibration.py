# DEV diagnostic script (not product code). Reads runs from /tmp/dev/runs and clips from the staged DEV folder.
import json, sys, numpy as np, glob, copy
from spana.config import load_config
from spana.video import VideoSource
from spana.artifacts import read_json, detections_from_frame
from spana.tracking.camera import CameraMotionEstimator
from spana.tracking.tracker import MultiPlayerTracker
from spana.appearance import describe, similarity
from spana.geometry import box_to_cxcywh, iou_matrix
cfg=load_config('config/default.yaml'); D='/mnt/user-data/uploads/Downloads/SPANA_fresh_clips/'
same={'sim':[], 'm2':[], 'ratio':[], 'dh':[]}; diff=[]
for name in sys.argv[1:]:
    f=glob.glob(D+'FRESH_'+name[1:]+'_*.mp4')[0]
    v=VideoSource(f); dd=read_json(f'/tmp/dev/runs/{name}.det.json'); dets={fr['frame']:fr for fr in dd['frames']}
    cam=CameraMotionEstimator(cfg.camera,v.meta.width,v.meta.height,v.meta.fps); trk=MultiPlayerTracker(cfg,v.meta.width,v.meta.height)
    stride=dd['frame_stride']; first=min(dets); prev=None
    for i,t,img in v.frames(first,max(dets)+1):
        if (i-first)%stride: continue
        ds=detections_from_frame(dets[i]); boxes=np.array([x.box for x in ds]).reshape(-1,4); desc=describe(img,list(map(tuple,boxes)),cfg.appearance)
        co=cam.observe(img,[tuple(b) for b in boxes])
        if prev is not None and co.status.value=='OK' and len(ds):
            p=copy.deepcopy(trk)
            for tr in p._tracks: tr.kf.apply_camera(co.matrix()); tr.kf.predict(t-tr.last_update_t)
            c=(boxes[:,:2]+boxes[:,2:])/2; hgt=boxes[:,3]-boxes[:,1]
            for tr in p._tracks:
                if not (tr.confirmed and tr.observed_last_frame and tr.appearance is not None): continue
                pc=tr.kf.x[:2]; ph=tr.kf.height; dist=np.hypot(*(c-pc).T)/ph
                order=np.argsort(dist)
                if dist[order[0]]>0.5 or (len(order)>1 and dist[order[1]]<1.5): continue   # isolated, unambiguous
                k=order[0]
                if ds[k].score<cfg.detection.score_high: continue
                s=similarity(desc[k],tr.appearance)
                if s is None: continue
                same['sim'].append(s); same['m2'].append(tr.kf.position_mahalanobis2(c[k])); same['ratio'].append(hgt[k]/ph); same['dh'].append(dist[k])
                # different-player similarity: the same track vs detections >= 1.5 h away
                for j in order[1:4]:
                    sj=similarity(desc[j],tr.appearance)
                    if sj is not None: diff.append(sj)
        trk.update(i,t,ds,co,desc); prev=i
for k,vv in same.items():
    a=np.array(vv); print(f'same-player {k:5s} n={len(a)} p1 {np.percentile(a,1):.3f} p5 {np.percentile(a,5):.3f} p50 {np.median(a):.3f} p95 {np.percentile(a,95):.3f} p99 {np.percentile(a,99):.3f}')
a=np.array(diff); print(f'diff-player sim   n={len(a)} p5 {np.percentile(a,5):.3f} p50 {np.median(a):.3f} p95 {np.percentile(a,95):.3f} p99 {np.percentile(a,99):.3f}')
print('share same-player sim < min_similarity(%.2f): %.3f'%(cfg.appearance.min_similarity, np.mean(np.array(same['sim'])<cfg.appearance.min_similarity)))
print('share same-player m2 > gate_chi2(%.2f): %.3f'%(cfg.tracking.gate_chi2, np.mean(np.array(same['m2'])>cfg.tracking.gate_chi2)))
