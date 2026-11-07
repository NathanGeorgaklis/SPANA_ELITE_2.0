# DEV diagnostic script (not product code). Reads runs from /tmp/dev/runs and clips from the staged DEV folder.
import json, sys, numpy as np, collections, glob
from spana.config import load_config
from spana.video import VideoSource
from spana.artifacts import read_json, detections_from_frame
from spana.tracking.camera import CameraMotionEstimator
from spana.tracking.tracker import MultiPlayerTracker
from spana.appearance import describe, similarity
from spana.geometry import box_to_cxcywh, iou_matrix
name=sys.argv[1]
cfg=load_config('config/default.yaml')
D='/mnt/user-data/uploads/Downloads/SPANA_fresh_clips/'
f=glob.glob(D+'FRESH_'+name[1:3]+'_*.mp4')[0]
v=VideoSource(f); dd=read_json(f'/tmp/dev/runs/'+name+'.det.json'); dets={fr['frame']:fr for fr in dd['frames']}
cam=CameraMotionEstimator(cfg.camera,v.meta.width,v.meta.height,v.meta.fps); trk=MultiPlayerTracker(cfg,v.meta.width,v.meta.height)
stride=dd['frame_stride']; first=min(dets)
cause=collections.Counter(); prevobs=set(); events=[]
for i,t,img in v.frames(first,max(dets)+1):
    if (i-first)%stride: continue
    ds=detections_from_frame(dets[i]); boxes=[x.box for x in ds]; desc=describe(img,boxes,cfg.appearance)
    co=cam.observe(img,boxes)
    # snapshot of tracks that were observed last frame, BEFORE update: compute after camera+predict inside update... replicate by deep copy
    import copy
    pre=copy.deepcopy(trk)
    ft=trk.update(i,t,ds,co,desc)
    obs_now={o.track_id for o in ft.tracks if o.detection_index is not None}
    for o in ft.tracks:
        if o.track_id in prevobs and o.detection_index is None and o.confirmed:
            # replay gates on pre-tracker after its own camera/predict step
            p=copy.deepcopy(pre)
            if co.status.value=='OK':
                for tr in p._tracks: tr.kf.apply_camera(co.matrix())
            for tr in p._tracks: tr.kf.predict(t-tr.last_update_t); tr.last_update_t=t
            tr=next(x for x in p._tracks if x.track_id==o.track_id)
            pc=tr.kf.x[:2]; ph=tr.kf.height
            best=None
            for k,dt_ in enumerate(ds):
                c=box_to_cxcywh(dt_.as_array())[:2]; dh=np.hypot(*(c-pc))/ph
                if best is None or dh<best[0]: best=(dh,k)
            if best is None or best[0]>1.0: cause['detector_miss']+=1; continue
            k=best[1]; dt_=ds[k]; c=box_to_cxcywh(dt_.as_array())[:2]
            m2=tr.kf.position_mahalanobis2(c); ratio=dt_.height/ph
            radius=p._gate_radius_h(tr,t); sim=similarity(desc[k],tr.appearance)
            iou=float(iou_matrix(tr.box()[None],dt_.as_array()[None])[0,0])
            taken=[x.track_id for x in ft.tracks if x.detection_index==k]
            r=[]
            if best[0]>radius: r.append('centre_gate')
            if m2>cfg.tracking.gate_chi2: r.append('mahalanobis')
            if not (1/cfg.tracking.max_height_ratio<=ratio<=cfg.tracking.max_height_ratio): r.append('height_ratio')
            if sim is not None and sim<cfg.appearance.min_similarity: r.append('appearance_veto')
            if dt_.score<cfg.detection.score_high and iou<cfg.tracking.low_score_min_iou: r.append('lowscore_iou')
            if taken: r.append('taken_by_'+('dup?' if taken else ''))
            cause[' + '.join(r) or 'other(refusal)']+=1
            events.append({'frame':i,'track':o.track_id,'cause':' + '.join(r),'box':[float(x) for x in tr.box()],'det':list(dt_.box),'taken':taken,'sim':sim,'ratio':float(ratio),'score':dt_.score})
    prevobs=obs_now
for k,v_ in cause.most_common(): print(f'{v_:4d}  {k}')
json.dump(events,open(f'/tmp/dev/{name}_loss_events.json','w'))
