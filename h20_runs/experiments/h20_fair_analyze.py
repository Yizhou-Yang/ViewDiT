#!/usr/bin/env python3
"""Analysis for the fairness round.
1) speed/PSNR/LPIPS per method from gen_*.jsonl + score_*.jsonl (fidelity to full)
2) optional reference-free metrics from refq_*.jsonl (aesthetic, clip_text, subj_cons, imaging, motion, flicker)
3) per speed tier, best of each FAMILY (family = prefix before first digit-ish knob), paired diff vs best OURS.
Usage: python3 h20_fair_analyze.py DIR PHASE [--select K]  -> writes FAIR_{PHASE}.md (+ fairsel json with --select)
"""
import glob, json, math, re, sys
from collections import defaultdict
import numpy as np

D, PH = sys.argv[1], sys.argv[2]
gen = [json.loads(l) for fn in glob.glob(f'{D}/gen_{PH}*.jsonl') for l in open(fn)]
sc = {(r['prompt'], r['seed'], r['method']): r for fn in glob.glob(f'{D}/score_{PH}*.jsonl') for r in map(json.loads, open(fn))}
rq = {(r['prompt'], r['seed'], r['method']): r for fn in glob.glob(f'{D}/refq_{PH}*.jsonl') for r in map(json.loads, open(fn))}
full_t = {(r['prompt'], r['seed']): r['seconds'] for r in gen if r['method'] == 'full'}


def family(m):
    if m.startswith('OURS'): return 'OURS'
    if m.startswith('full'): return m
    for p in ('TEAw', 'TEA_', 'FORAw', 'FORA_', 'TSw', 'TS_', 'Oonly', 'PAB', 'FC', 'dpm3_', 'dpm', 'unipc', 'steps'):
        if m.startswith(p):
            f = p.rstrip('_')
            return f + ('_GI' if m.endswith('_GI') and p == 'TEAw' else '')
    return m


per = defaultdict(lambda: defaultdict(dict))
for r in gen:
    k = (r['prompt'], r['seed'])
    if k not in full_t: continue
    e = per[r['method']][k]
    e['speed'] = full_t[k] / r['seconds']
    e['lmse'] = r['latent_mse_to_full']
    s = sc.get((*k, r['method']))
    if s and s.get('psnr_to_full') is not None:
        e['psnr'] = s['psnr_to_full']; e['lpips'] = s['lpips_to_full']; e['ssim'] = s['ssim_to_full']
    q, qf = rq.get((*k, r['method'])), rq.get((*k, 'full'))
    if q and qf:
        for m in ('aesthetic', 'clip_text', 'subj_cons', 'imaging', 'motion', 'flicker'):
            e['d_' + m] = q[m] - qf[m] if m not in ('imaging', 'motion', 'flicker') else q[m] / max(qf[m], 1e-12)


def mean(m, key):
    v = [e[key] for e in per[m].values() if key in e]
    return (float(np.mean(v)), len(v)) if v else (float('nan'), 0)


rows = []
for m in per:
    sp, n = mean(m, 'speed')
    r = dict(method=m, fam=family(m), n=n, speed=sp)
    for key in ('psnr', 'lpips', 'ssim', 'lmse', 'd_aesthetic', 'd_clip_text', 'd_subj_cons', 'd_imaging', 'd_motion', 'd_flicker'):
        r[key] = mean(m, key)[0]
    r['psnr_lmse'] = -10 * math.log10(max(r['lmse'], 1e-12))
    rows.append(r)
key = 'psnr' if any(not math.isnan(r['psnr']) for r in rows) else 'psnr_lmse'
tiers = [(1.3, 1.65), (1.65, 1.95), (1.95, 2.25), (2.25, 2.7), (2.7, 3.1), (3.1, 3.6), (3.6, 5.0)]
L = [f'# Fairness round `{PH}` in {D}', '', f'quality key = {key} (psnr_lmse = -10log10 latent MSE to full, used when pixel scores missing)', '',
     '## All methods', '', '| method | fam | n | speed | PSNR | LPIPS | latPSNR | dAesth | dCLIP | dSubj | imaging× | motion× | flicker× |', '|' + '---|' * 13]
f = lambda x, d=2: 'nan' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'
for r in sorted(rows, key=lambda r: -r['speed']):
    L.append(f"| {r['method']} | {r['fam']} | {r['n']} | {f(r['speed'])} | {f(r['psnr'])} | {f(r['lpips'],3)} | {f(r['psnr_lmse'])} | {f(r['d_aesthetic'],3)} | "
             f"{f(r['d_clip_text'])} | {f(r['d_subj_cons'],4)} | {f(r['d_imaging'],3)} | {f(r['d_motion'],3)} | {f(r['d_flicker'],3)} |")
L += ['', '## Best of each family per speed tier (by quality key)', '']
sel = set()
for lo, hi in tiers:
    best = {}
    for r in rows:
        if lo <= r['speed'] < hi and not math.isnan(r[key]) and not r['fam'].startswith('full'):
            if r['fam'] not in best or r[key] > best[r['fam']][key]:
                best[r['fam']] = r
    if not best: continue
    L.append(f'### {lo}-{hi}x')
    for r in sorted(best.values(), key=lambda r: -r[key]):
        L.append(f"- {r['method']} ({r['fam']}): {r['speed']:.2f}x, {key} {r[key]:.2f}, LPIPS {f(r['lpips'],3)}, dAesth {f(r['d_aesthetic'],3)}, dCLIP {f(r['d_clip_text'])}")
        sel.add(r['method'])
    L.append('')
open(f'{D}/FAIR_{PH}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
if '--select' in sys.argv:
    out = sys.argv[sys.argv.index('--select') + 1]
    sel |= {'full_perturb1e-2'}
    json.dump(sorted(sel), open(out, 'w'))
    print('selected', len(sel), sorted(sel))
