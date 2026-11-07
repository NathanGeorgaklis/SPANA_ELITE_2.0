import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import numpy as np
from spana.config import load_config
from spana.detection.tiling import detect_tiled, tile_grid
CFG = load_config("/home/claude/spana_elite_2.0/config/default.yaml").detection
W, H = 1920, 1080
grid = tile_grid(W, H, CFG); print("grid", grid)
def make(people, inset):
    calls = []
    def infer(img):
        h, w = img.shape[:2]
        if (h, w) == (H, W): ox = oy = 0
        else: ox, oy, _ = grid[len(calls)]; calls.append(1)
        out = []
        for b in people:
            x1, y1 = max(0, b[0]-ox), max(0, b[1]-oy); x2, y2 = min(w, b[2]-ox), min(h, b[3]-oy)
            if x2 <= x1 or y2 <= y1: continue
            # detector regresses a truncated edge `inset` px inside the crop edge
            if b[0]-ox < 0: x1 += inset
            if b[1]-oy < 0: y1 += inset
            if b[2]-ox > w: x2 -= inset
            if b[3]-oy > h: y2 -= inset
            if (y2-y1)*640/max(h,w) >= 12: out.append([x1,y1,x2,y2])
        return np.array(out,float).reshape(-1,4), np.full(len(out),0.9)
    return infer
frame = np.zeros((H,W,3),np.uint8)
# person straddling vertical tile boundary x=648 (tile0 right internal edge) and x=486 (tile1 left)
cases = {"straddle_x_small(60x150)": [[620,500,680,650]],
         "straddle_y_internal(40x100) at y=648": [[300,600,340,700]],
         "large_body(120x400) across tiles": [[600,300,720,700]]}
for name, ppl in cases.items():
    for inset in (0, 5, 10, 20):
        d = detect_tiled(make(ppl, inset), frame, CFG, 0)
        print(f"{name:38s} inset={inset:2d}px -> {len(d)} dets for {len(ppl)} person:", [tuple(round(v) for v in x.box) for x in d])
