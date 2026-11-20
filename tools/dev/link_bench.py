"""DEV bench for offline tracklet linking (not product code): which identity transfers did the links cause?

    python -m tools.dev.link_bench transfers --out OUT.json \
        --clip NAME BASE.bench.json LINKED.bench.json LINKED.links.jsonl [--clip ...]

tools/benchmark.py counts a transfer as flagged when its track has `recoveries > 0`, and every linked track has one,
so a wrong link would never show there as silent. Here a transfer is LINK-CAUSED when it appears in the linked run's
benchmark events but not in the baseline's (same frame, from-player, to-player). A link-caused transfer counts as
SILENT unless the track was IDENTITY_UNCERTAIN (now or since its previous attribution): that is how the follower
sees it. Link log: links and abstentions by reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List


def _key(e: dict) -> tuple:
    return e["frame"], e["from"], e["to"]


def clip_transfers(base: dict, linked: dict, links: List[dict]) -> dict:
    b = Counter(_key(e) for e in base["events"]["transfers"])
    new, gone = [], Counter(b)
    for e in linked["events"]["transfers"]:
        k = _key(e)
        if gone[k] > 0:
            gone[k] -= 1
        else:
            new.append(e)
    rows = [r for r in links if r.get("kind") in ("link", "abstain")]
    reasons = Counter(x for r in rows if r["kind"] == "abstain" for x in r["reasons"])
    return {"links": sum(r["kind"] == "link" for r in rows), "abstentions": sum(r["kind"] == "abstain" for r in rows),
            "links_cleared_by_appearance": sum(r["kind"] == "link" and r["cleared_by_appearance"] for r in rows),
            "abstention_reasons": dict(sorted(reasons.items())),
            "link_caused_transfers": len(new),
            "link_caused_silent": sum(not e["identity_uncertain"] for e in new),
            "link_caused_events": new,
            "baseline_transfers_removed": sum(gone.values())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("transfers")
    s.add_argument("--clip", nargs=4, action="append", required=True, metavar=("NAME", "BASE", "LINKED", "LINKS"))
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    clips, pooled = {}, Counter()
    for name, base, linked, links in a.clip:
        rows = [json.loads(l) for l in Path(links).read_text().splitlines() if l.strip()]
        c = clip_transfers(json.loads(Path(base).read_text()), json.loads(Path(linked).read_text()), rows)
        clips[name] = c
        for k in ("links", "abstentions", "links_cleared_by_appearance", "link_caused_transfers",
                  "link_caused_silent", "baseline_transfers_removed"):
            pooled[k] += c[k]
        for k, v in c["abstention_reasons"].items():
            pooled["reason:" + k] += v
    doc = {"clips": clips, "pooled": dict(sorted(pooled.items()))}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(doc["pooled"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
