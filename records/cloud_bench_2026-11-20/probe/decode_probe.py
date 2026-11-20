"""DEV probe: fetch a few decoded frames from the Modal container and compare them with the Mac decode."""
import io, sys, tempfile
from pathlib import Path
import numpy as np, modal
WT = Path(sys.argv[1]); sys.path.insert(0, str(WT))
from tools import cloud_bench as cb
cfg = cb.load_bench_config(WT/"config/cloud_bench.yaml", WT/"config/roboflow.yaml")
f = cfg.footage_by_name("FRESH_02")
with tempfile.TemporaryDirectory() as td:
    cb.stage_code(Path(td))
    app = modal.App("spana-decode-probe")
    vol = modal.Volume.from_name(cfg.volume_name)
    def grab(path):
        import sys as s; s.path.insert(0, "/root/repo")
        import io as i, numpy as n, cv2
        from spana.video import VideoSource
        fr = [x for _, _, x in VideoSource(path).frames(0, 3)]
        b = i.BytesIO(); n.savez_compressed(b, *fr)
        return {"npz": b.getvalue(), "cv2": cv2.__version__, "build": [l for l in cv2.getBuildInformation().splitlines() if "FFMPEG" in l or "avcodec" in l or "swscale" in l]}
    fn = app.function(image=cb._image(cfg, Path(td)), volumes={cb.VOL: vol}, serialized=True, timeout=600)(grab)
    with app.run():
        r = fn.remote(f"/vol/footage/{f.name}/{f.file}")
import cv2
from spana.video import VideoSource
loc = [x for _, _, x in VideoSource(str(Path.home()/"Downloads/SPANA_fresh_clips"/f.file)).frames(0, 3)]
rem = np.load(io.BytesIO(r["npz"]))
print("cloud", r["cv2"], r["build"]); print("mac", cv2.__version__, [l for l in cv2.getBuildInformation().splitlines() if "FFMPEG" in l or "avcodec" in l or "swscale" in l])
for k, a in zip(rem.files, loc):
    b = rem[k]; d = np.abs(b.astype(int) - a.astype(int))
    print(k, b.shape, a.shape, "max", d.max(), "mean", round(d.mean(), 4), "share>0", round((d > 0).mean(), 4), "share>2", round((d > 2).mean(), 5))
