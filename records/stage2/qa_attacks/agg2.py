import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from collections import Counter
import adv, diag
c = Counter()
for name in adv.S:
    if not name.startswith("crowd") or "none50" in name: continue
    for fps in adv.FPS:
        for x in diag.diag(name, fps):
            if x[5] != "S1": continue
            kp, kg = int(x[3][1:]) % 2, int(x[4][1:]) % 2
            c["diff_kit" if kp != kg else "same_kit"] += 1
print(c)
