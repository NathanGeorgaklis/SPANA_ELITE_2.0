import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from collections import Counter
import adv, diag
c = Counter(); n=0; dists=[]
for name in adv.S:
    if not name.startswith("crowd"): continue
    for fps in adv.FPS:
        for x in diag.diag(name, fps):
            n+=1; c[(x[5], x[6], x[7])]+=1
            if x[9] is not None: dists.append(x[9])
print("transfers", n); print(c)
import numpy as np; print("dist_h of transferring match: median %.3f max %.3f" % (np.median(dists), max(dists)))
