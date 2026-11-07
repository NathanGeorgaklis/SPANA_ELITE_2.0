# DEV diagnostic script (not product code). Reads runs from /tmp/dev/runs and clips from the staged DEV folder.
import json, collections, sys
n=sys.argv[1]
t=json.load(open(f'/tmp/dev/runs/{n}.trk.json'))
cams=collections.Counter(fr['camera']['status'] for fr in t['frames'])
rows=t['tracklets']; conf=[r for r in rows if r['confirmed']]
ends=collections.Counter(r['end_reason'] for r in rows)
live=[sum(1 for x in fr['tracks'] if x['source']=='OBSERVED' and x['confirmed']) for fr in t['frames']]
dec=collections.Counter()
for l in open(f'/tmp/dev/runs/{n}.trk.decisions.jsonl'):
    d=json.loads(l); dec[(d['kind'],d.get('reason'))]+=1
print(n,'frames',len(t['frames']),'camera',dict(cams)); print(' tracklets',len(rows),'confirmed',len(conf),'ends',dict(ends))
print(' observed confirmed per frame: median',sorted(live)[len(live)//2],'min',min(live),'max',max(live))
print(' decisions',{f'{k[0]}:{k[1]}':v for k,v in dec.items() if k[0] not in ('match','camera')})
