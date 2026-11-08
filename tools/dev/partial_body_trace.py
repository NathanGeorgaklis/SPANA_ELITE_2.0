# DEV diagnostic script (not product code). For a track change caused by a box-extent jump, trace the old track
# back over its last observed frames: the detection it matched (size, score, flag), the labelled full-body box of
# the player (nearest labelled frame, mapped to the track's current height scale only for comparison), and every
# other detection on the player at that frame with its owner (which track matched it, or spawn / refusal).
#   python tools/dev/partial_body_trace.py LABEL_DIR DETECTIONS TRACKS DECISIONS FRAME:PLAYER:TRACK [...]
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from tools.benchmark import load_frozen  # noqa: E402

BACK = 12          # processed frames to trace back


def ioa(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def fmt(b):
    return f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] h{b[3]-b[1]:.0f} w{b[2]-b[0]:.0f}"


def main(label_dir, det_path, trk_path, dec_path, *cases):
    fz, gt = load_frozen(Path(label_dir))
    dets = {fr["frame"]: fr["detections"] for fr in json.load(open(det_path))["frames"]}
    trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in json.load(open(trk_path))["frames"]}
    dec = collections.defaultdict(list)
    for line in open(dec_path):
        d = json.loads(line)
        dec[d["frame"]].append(d)
    frames = sorted(dets)
    for case in cases:
        f, g, t0 = case.split(":"); f = int(f)
        lab = min((x for x in gt.players if g in gt.players[x]), key=lambda x: abs(x - f))
        G = gt.players[lab][g]
        print(f"\n=== frame {f} player {g} old track {t0}; nearest label frame {lab}: GT {fmt(G)}")
        idx = frames.index(f)
        for fr in frames[max(0, idx - BACK): idx + 2]:
            owner = {}
            for d in dec[fr]:
                if d["kind"] in ("match", "recovery") and d.get("detection") is not None:
                    owner[d["detection"]] = f"{d['kind']}:{d.get('stage', '')}:{d['track_id']}"
                elif d["kind"] == "spawn":
                    owner[d["detection"]] = f"spawn:{d['track_id']}"
                elif d["kind"] == "spawn_refused":
                    owner.setdefault(d["detection"], f"spawn_refused:{d['reason']}")
            t = trk[fr].get(t0)
            tb = t["box"] if t else None
            line = f"  f{fr} {t0}: " + (f"{t['state'][:3]} {t['source'][:3]} det={t['detection_index']} {fmt(tb)}" if t else "—")
            print(line)
            ref = tb or G
            for i, r in enumerate(dets[fr]):
                b = r[:4]
                if ioa(b, ref) < 0.3 and ioa(ref, b) < 0.3:
                    continue
                print(f"      d{i:<3} s{r[4]:.2f} mb{r[6]} {fmt(b)} ioa(track in det)={ioa(ref, b):.2f} "
                      f"ioa(det in track)={ioa(b, ref):.2f} -> {owner.get(i, 'unused')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
