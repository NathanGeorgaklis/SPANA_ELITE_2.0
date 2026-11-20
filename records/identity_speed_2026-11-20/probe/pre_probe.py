import time, numpy as np, cv2, sys
from spana.detection.rfdetr_local import preprocess, MEAN, STD
from spana.video import VideoSource
def fast(frame_bgr, resolution):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    u = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    out = np.empty((1, 3, resolution, resolution), np.float32)
    for c in range(3):
        p = out[0, c]
        np.divide(u[:, :, c], np.float32(255.0), out=p, dtype=np.float32)
        np.subtract(p, MEAN[c], out=p)
        np.divide(p, STD[c], out=p)
    return out
print("numpy", np.__version__)
src = VideoSource(sys.argv[1]); n=0; same=True
for i,t,img in src.frames(45000, 45060):
    a, b = preprocess(img,1280), fast(img,1280)
    same &= a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(); n+=1
rng=np.random.default_rng(3)
for _ in range(5):
    img=(rng.random((720,1280,3))*256).astype(np.uint8)
    same &= preprocess(img,1280).tobytes()==fast(img,1280).tobytes()
print("frames", n, "bit-identical", same)
for f in (preprocess, fast):
    t=time.perf_counter()
    for _ in range(30): f(img,1280)
    print(f.__name__, (time.perf_counter()-t)/30*1e3)
