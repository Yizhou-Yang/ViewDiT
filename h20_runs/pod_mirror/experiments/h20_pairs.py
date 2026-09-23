#!/usr/bin/env python3
"""Paired bootstrap over prompts for arbitrary method pairs: latent MSE (relative), PSNR (dB), sharpness/motion (% vs full), seconds.
usage: h20_pairs.py ROOT A:B [A:B ...]   (positive = A better for mse/psnr)"""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
ev, sc = {}, {}
for f in root.glob('eval/eval_shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); ev[(r['prompt'], r['seed'], r['method'])] = r
for f in root.glob('score/score_shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); sc[(r['prompt'], r['seed'], r['method'])] = r

def boot(d, n=4000):
    rnd = random.Random(0); ps = list(d)
    ms = sorted(statistics.mean(statistics.mean(d[p]) for p in (rnd.choice(ps) for _ in ps)) for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n)]

def fmt(d):
    if len(d) < 4:
        return 'n/a'
    v = [x for xs in d.values() for x in xs]; lo, hi = boot(d)
    star = '*' if lo > 0 or hi < 0 else ' '
    return f'{statistics.mean(v):+7.3f} [{lo:+7.3f},{hi:+7.3f}]{star} n={len(v)}'

out = {}
for pair in sys.argv[2:]:
    A, B = pair.split(':')
    m, p, sh, mo, t = (defaultdict(list) for _ in range(5))
    for (pr, s, meth), r in ev.items():
        if meth != A or (pr, s, B) not in ev:
            continue
        rb = ev[(pr, s, B)]
        m[pr].append((rb['latent_mse_to_full'] - r['latent_mse_to_full']) / max(rb['latent_mse_to_full'], 1e-12) * 100)
        t[pr].append(r['seconds'] - rb['seconds'])
        ka, kb, kf = (pr, s, A), (pr, s, B), (pr, s, 'full')
        if ka in sc and kb in sc and kf in sc:
            p[pr].append(sc[ka]['psnr_to_full'] - sc[kb]['psnr_to_full'])
            f = sc[kf]
            sh[pr].append((sc[ka]['sharpness'] - sc[kb]['sharpness']) / f['sharpness'] * 100)
            mo[pr].append((sc[ka]['motion_magnitude'] - sc[kb]['motion_magnitude']) / f['motion_magnitude'] * 100)
    row = dict(mse_rel_gain_pct=fmt(m), psnr_db=fmt(p), sharp_pct=fmt(sh), motion_pct=fmt(mo), dsec=fmt(t))
    out[pair] = row
    print(f'== {A} vs {B}')
    for k, v in row.items():
        print(f'   {k:18s} {v}')
Path(root / 'PAIRS_last.json').write_text(json.dumps(out, indent=1))
