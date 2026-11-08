# DEV diagnostic script (not product code). Relates candidate camera-uncertainty measures (inlier RMS, standard error,
# all-feature RMS disagreement) to the common-mode innovation of matched tracks, per frame, binned by inlier count.
# usage: python tools/dev/camera_disagreement_calibration.py VIDEO CONFIG DETECTIONS OUT.npy
import sys, collections, numpy as np, cv2
sys.path.insert(0, '.')
import tools.dev.motion_gate_probe as M
from spana.tracking import camera as C
from spana.geometry import box_to_cxcywh
CAP = {}
_est = cv2.estimateAffinePartial2D
cur = {"f": None}
def est(src, dst, **kw):
    A, inl = _est(src, dst, **kw)
    if A is not None and inl is not None:
        m = inl.reshape(-1) == 1
        s, d = src.reshape(-1, 2)[m], dst.reshape(-1, 2)[m]
        pred = s @ A[:, :2].T + A[:, 2]
        r = d - pred
        allr = dst.reshape(-1, 2) - (src.reshape(-1, 2) @ A[:, :2].T + A[:, 2])
        CAP[cur["f"]] = dict(n=int(m.sum()), ntot=len(m), rstd=float(np.sqrt((r ** 2).mean())),
                             spread=float(np.hypot(*s.std(0))), allmed=float(np.median(np.hypot(*allr.T))), rms_all=float(np.sqrt((allr**2).sum(1).mean())), out_vec=float(np.hypot(*np.median(allr[~m],0))) if (~m).any() else 0.0)
    return A, inl
C.cv2.estimateAffinePartial2D = est
PX = collections.defaultdict(list)
_orig = M._apply
def app(self, matched, dets, desc, fi, t):
    for k, (i, _) in matched.items():
        tr = self._tracks[k]
        if tr.observed_last_frame and tr.confirmed:
            PX[self._frame].append(box_to_cxcywh(dets[i].as_array())[:2] - tr.kf.x[:2])
    return _orig(self, matched, dets, desc, fi, t)
M._apply = app
# hook frame index into estimator: wrap observe
_obs = C.CameraMotionEstimator.observe
FR = {"i": 0}
def obs(self, img, boxes):
    cur["f"] = FR["i"]; return _obs(self, img, boxes)
C.CameraMotionEstimator.observe = obs
_upd = M.T.MultiPlayerTracker.update
def upd(self, i, *a, **k):
    return _upd(self, i, *a, **k)
# frame counter via video.frames wrapper
from spana import video as V
_frames = V.VideoSource.frames
def frames(self, a, b):
    for i, t, img in _frames(self, a, b):
        FR["i"] = i; yield i, t, img
V.VideoSource.frames = frames
M.T.MultiPlayerTracker._gate_report = M._orig_gate
cfg, ddoc, fr, log = M.replay(*sys.argv[1:4])
M.T.MultiPlayerTracker._apply_matches = app
k = cfg.camera.analysis_width_px / 1920
rows = []
for f, v in PX.items():
    if len(v) >= 2 and f in CAP:
        c = CAP[f]; cm = float(np.hypot(*np.median(np.array(v), 0)))
        rows.append((c['n'], c['rstd'] / k, c['rstd'] / k / np.sqrt(c['n']), cm, c['allmed'] / k, c['spread'] / k, c['n']/c['ntot'], f, len(v), c['rms_all']/k, c['out_vec']/k))
a = np.array(rows)
for lo, hi in ((25, 50), (50, 100), (100, 200), (200, 1000)):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if m.any():
        print(f"inl {lo}-{hi}: frames {m.sum():4d} resid_rms_px {np.median(a[m,1]):.2f} se_px {np.median(a[m,2]):.3f} common_mode_px med {np.median(a[m,3]):.2f} p90 {np.percentile(a[m,3],90):.2f} spread_px {np.median(a[m,5]):.0f}")
    if m.any(): print("    inlier_frac med", round(float(np.median(a[m,6])),3), "all-feature median resid px", round(float(np.median(a[m,4])),2))
print("corr(common_mode, se)", np.corrcoef(a[:,3], a[:,2])[0,1], "corr(common_mode, resid)", np.corrcoef(a[:,3], a[:,1])[0,1], "corr(cm,1/sqrt n)", np.corrcoef(a[:,3], 1/np.sqrt(a[:,0]))[0,1])
np.save(sys.argv[4], a)
for name, col in (("rms_all", 9), ("out_vec*out_frac", None), ("inlier_rms", 1)):
    sig = a[:, col] if col is not None else a[:, 10] * (1 - a[:, 6])
    for lo, hi in ((25, 50), (50, 100), (100, 10000)):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.any():
            r = a[m, 3] / np.maximum(sig[m], 1e-3)
            print(f"{name:18s} inl {lo}-{hi}: sigma_px med {np.median(sig[m]):6.2f}  cm/sigma p50 {np.median(r):6.2f} p90 {np.percentile(r,90):6.2f}")
