import time, threading, numpy as np
from spana.detection.rfdetr_local import LocalModelRecord, make_runner, preprocess
rec = LocalModelRecord.load("models/rfdetr_v1.json")
run = make_runner(rec, "torch", "mps", "fp32", True)
img = (np.random.default_rng(0).random((720,1280,3))*255).astype(np.uint8)
x = preprocess(img, rec.resolution)
for _ in range(3): run(x)
N=20
t=time.perf_counter()
for _ in range(N): run(x)
m=(time.perf_counter()-t)/N
t=time.perf_counter()
for _ in range(N): preprocess(img, rec.resolution)
p=(time.perf_counter()-t)/N
t=time.perf_counter()
for _ in range(N): preprocess(img, rec.resolution); run(x)
seq=(time.perf_counter()-t)/N
done=threading.Event()
def worker():
    for _ in range(N): preprocess(img, rec.resolution)
th=threading.Thread(target=worker); t=time.perf_counter(); th.start()
for _ in range(N): run(x)
th.join(); par=(time.perf_counter()-t)/N
print(f"model {m*1e3:.1f} pre {p*1e3:.1f} seq {seq*1e3:.1f} overlapped {par*1e3:.1f}")
