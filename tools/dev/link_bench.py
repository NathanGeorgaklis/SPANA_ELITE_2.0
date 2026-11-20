"""DEV bench for offline tracklet linking (not product code): thin wrapper over tools/benchmark.py.

    python -m tools.dev.link_bench summary --out OUT.json --clip NAME LINKED.bench.json LINKED.links.jsonl [--clip ...]

Transfer counting lives in spana.evaluation (tools/benchmark.py, 2026-11-20 rule): this only pools the benchmark's
link / recovery columns per clip and adds the link log's link / abstention counts by reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BENCH_KEYS = ("identity_transfers", "unflagged_transfers", "link_caused_transfers", "link_caused_silent",
              "recovery_caused_transfers", "recovery_caused_silent", "transfers_on_linked_tracks")


def clip_summary(bench: dict, links: list) -> dict:
    s = bench["summary"]
    if s.get("link_caused_transfers") is None:
        raise SystemExit("bench report predates the 2026-11-20 transfer rule: re-score it with tools/benchmark.py")
    rows = [r for r in links if r.get("kind") in ("link", "abstain")]
    return {**{k: s[k] for k in BENCH_KEYS},
            "links": sum(r["kind"] == "link" for r in rows), "abstentions": sum(r["kind"] == "abstain" for r in rows),
            "links_cleared_by_appearance": sum(r["kind"] == "link" and r["cleared_by_appearance"] for r in rows),
            "abstention_reasons": dict(sorted(Counter(x for r in rows if r["kind"] == "abstain"
                                                      for x in r["reasons"]).items()))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--clip", nargs=3, action="append", required=True, metavar=("NAME", "LINKED_BENCH", "LINKS"))
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    clips, pooled = {}, Counter()
    for name, bench, links in a.clip:
        rows = [json.loads(l) for l in Path(links).read_text().splitlines() if l.strip()]
        c = clip_summary(json.loads(Path(bench).read_text()), rows)
        clips[name] = c
        for k, v in c.items():
            if k == "abstention_reasons":
                for r, n in v.items():
                    pooled["reason:" + r] += n
            else:
                pooled[k] += v
    doc = {"clips": clips, "pooled": dict(sorted(pooled.items()))}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(doc["pooled"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
