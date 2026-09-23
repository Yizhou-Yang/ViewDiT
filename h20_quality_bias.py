#!/usr/bin/env python3
"""Paired bootstrap (over prompts) of per-video quality deltas vs full: motion, sharpness, subject, background."""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/data/viewdit/results')
sc = {}
for f in sorted(root.glob('score/score_shard*.jsonl')):
    for l in open(f):
        if l.strip():
            r = json.loads(l); sc[(r['prompt'], r['seed'], r['method'])] = r
methods = sorted({k[2] for k in sc} - {'full'})
Q = ['motion_magnitude', 'sharpness', 'subject_consistency', 'background_consistency']

def boot(d, n=4000):
    rnd = random.Random(0); ps = list(d)
    ms = sorted(statistics.mean(statistics.mean(d[p]) for p in (rnd.choice(ps) for _ in ps)) for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n)]

out = {}
print(f"{'method':22s} " + ' '.join(f'{q[:10]:>26s}' for q in Q))
for m in methods:
    row, cells = {}, []
    for q in Q:
        d = defaultdict(list)
        for (p, s, mm), r in sc.items():
            if mm == m and (p, s, 'full') in sc:
                f = sc[(p, s, 'full')][q]
                v = (r[q] - f) / f * 100 if q in ('motion_magnitude', 'sharpness') else (r[q] - f) * 100
                d[p].append(v)
        if len(d) < 4:
            cells.append(' ' * 26); continue
        allv = [v for vs in d.values() for v in vs]
        lo, hi = boot(d)
        row[q] = dict(mean=statistics.mean(allv), ci95=[lo, hi], n=len(allv))
        cells.append(f'{statistics.mean(allv):+6.2f} [{lo:+6.2f},{hi:+6.2f}]')
    out[m] = row
    print(f'{m:22s} ' + ' '.join(f'{c:>26s}' for c in cells))
(root / 'QUALITY_DELTAS.json').write_text(json.dumps(out, indent=2))
print('units: motion/sharpness = % vs full; subject/background = x100 absolute delta vs full')
