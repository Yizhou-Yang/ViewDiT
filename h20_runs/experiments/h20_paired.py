#!/usr/bin/env python3
"""Paired per-(prompt,seed) comparison OURS vs competitor at matched speed. usage: h20_paired.py OUTDIR phase"""
import glob, json, sys
import numpy as np

d, ph = sys.argv[1], sys.argv[2]
S = {}
for fn in glob.glob(f'{d}/score_{ph}*.jsonl'):
    for r in map(json.loads, open(fn)):
        S[(r['prompt'], r['seed'], r['method'])] = r
PAIRS = [('OURS_Ow20_L4', 'TEA_0.1'), ('OURS_Ow17_L4_GI', 'TEA_0.15'), ('OURS_O3w13_B2', 'TEA_0.2'),
         ('OURS_O3w13_B2', 'TS_n3'), ('OURS_O4w13_B2_GI', 'TEA_0.2'), ('OURS_O3w10_B3', 'TEA_0.3'),
         ('OURS_Ow20_L4', 'PAB_k3'), ('OURS_Ow20_L4', 'FC_v2'), ('OURS_O3w13_B2', 'steps20')]
rng = np.random.default_rng(0)
out = ['| ours | competitor | n | dPSNR mean | 95% CI | win rate | dLPIPS mean |', '|---|---|---|---|---|---|---|']
for a, b in PAIRS:
    keys = sorted({k[:2] for k in S if k[2] == a} & {k[:2] for k in S if k[2] == b})
    if not keys:
        continue
    dp = np.array([S[(*k, a)]['psnr_to_full'] - S[(*k, b)]['psnr_to_full'] for k in keys])
    dl = np.array([S[(*k, a)]['lpips_to_full'] - S[(*k, b)]['lpips_to_full'] for k in keys])
    bs = [rng.choice(dp, len(dp)).mean() for _ in range(5000)]
    lo, hi = np.quantile(bs, [.025, .975])
    out.append(f'| {a} | {b} | {len(dp)} | {dp.mean():+.2f} | [{lo:+.2f},{hi:+.2f}] | {(dp > 0).mean():.2f} | {dl.mean():+.4f} |')
txt = '\n'.join(out)
print(txt)
open(f'{d}/PAIRED_{ph}.md', 'w').write(txt + '\n')
