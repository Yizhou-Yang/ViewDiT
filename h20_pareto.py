#!/usr/bin/env python3
"""Pareto / same-speed-tier comparison from h20 summary json. usage: h20_pareto.py summary_bench.json [out.md]"""
import json, sys, math

S = json.load(open(sys.argv[1]))
fam = lambda k: ('OURS' if k.startswith('OURS') else 'TeaCache' if k.startswith('TEA') else 'PAB' if k.startswith('PAB')
                 else 'FORA' if k.startswith('FORA') else 'TaylorSeer' if k.startswith('TS_') else 'FasterCache' if k.startswith('FC')
                 else 'steps' if k.startswith('steps') else 'ref')
rows = [(k, v) for k, v in S.items() if k != 'full' and not math.isnan(v.get('psnr', float('nan')))]
L = ['| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |', '|---|---|---|---|---|---|---|']
tiers = [(1.3, 1.7), (1.7, 2.05), (2.05, 2.45), (2.45, 2.85), (2.85, 3.2), (3.2, 4.0)]
out = []
for lo, hi in tiers:
    out.append(f'\n### speed {lo}-{hi}x\n'); out += L
    best = {}
    for k, v in rows:
        if lo <= v['speed'] < hi:
            f = fam(k)
            if f not in best or v['psnr'] > best[f][1]['psnr']:
                best[f] = (k, v)
    for f, (k, v) in sorted(best.items(), key=lambda x: -x[1][1]['psnr']):
        out.append(f"| {f} | {k} | {v['speed']:.2f} | {v['psnr']:.2f} | {v['ssim']:.4f} | {v['lpips']:.4f} | {v['dclip']:+.2f} |")
# pareto front per family
out.append('\n### Pareto front per family (speed, PSNR)\n')
for f in sorted({fam(k) for k, _ in rows}):
    pts = sorted([(v['speed'], v['psnr'], k) for k, v in rows if fam(k) == f], reverse=True)
    front, bestp = [], -1
    for s, p, k in pts:
        if p > bestp:
            front.append(f'{k}({s:.2f}x,{p:.1f})'); bestp = p
    out.append(f'- {f}: ' + ', '.join(front))
txt = '\n'.join(out)
print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], 'w').write(txt + '\n')
