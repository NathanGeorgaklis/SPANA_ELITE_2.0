import time, threading, numpy as np, cv2
from spana.detection.rfdetr_local import LocalModelRecord, make_runner, preprocess, MEAN, STD
rec = LocalModelRecord.load("models/rfdetr_v1.json")
run = make_runner(rec, "torch", "mps", "fp32", True)
img = (np.random.default_rng(0).random((720,1280,3))*255).astype(np.uint8)
x = preprocess(img, 1280)
for _ in range(3): run(x)
N=15
def timeit(f, n=N):
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e3
# pieces of preprocess
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); r=cv2.resize(rgb,(1280,1280))
print("cvt+resize", timeit(lambda: cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(1280,1280))))
print("astype/255", timeit(lambda: r.astype(np.float32)/255.0))
f=r.astype(np.float32)/255.0
print("norm", timeit(lambda: (f-MEAN)/STD))
n=(f-MEAN)/STD
print("transpose contig", timeit(lambda: np.ascontiguousarray(n.transpose(2,0,1)[None])))
# GIL during model: python counter in thread
cnt=[0]; stop=threading.Event()
def spin():
    while not stop.is_set(): cnt[0]+=1
th=threading.Thread(target=spin); th.start(); time.sleep(0.5); stop.set(); th.join(); idle=cnt[0]*2
cnt[0]=0; stop.clear(); th=threading.Thread(target=spin); th.start()
t=time.perf_counter(); 
for _ in range(5): run(x)
el=time.perf_counter()-t; stop.set(); th.join()
print(f"spin rate during model / idle: {cnt[0]/el/idle:.2f}; model with spinner {el/5*1e3:.1f}")
