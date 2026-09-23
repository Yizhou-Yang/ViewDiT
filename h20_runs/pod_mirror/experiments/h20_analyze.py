#!/usr/bin/env python3
"""Aggregate H20 cache-corrector study. Paired per (prompt, seed); bootstrap over prompts."""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/data/viewdit/results')

def load(pattern):
    rows = []
    for f in sorted(root.glob(pattern)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows

ev = load('eval/eval_shard*.jsonl')
sc = {(r['prompt'], r['seed'], r['method']): r for r in load('score/score_shard*.jsonl')}
T = defaultdict(dict)
for r in ev:
    T[r['method']][(r['prompt'], r['seed'])] = r
methods = sorted(T, key=lambda m: (m.split('_')[0], m))
keys = sorted(T['full'])

def boot(diff_by_prompt, n=4000, seed=0):
    rnd = random.Random(seed); ps = list(diff_by_prompt)
    means = sorted(statistics.mean(statistics.mean(diff_by_prompt[p]) for p in (rnd.choice(ps) for _ in ps)) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]

summary = {'n_videos_full': len(keys), 'methods': {}}
for m in methods:
    ks = [k for k in keys if k in T[m]]
    if not ks:
        continue
    row = dict(n=len(ks), seconds=statistics.mean(T[m][k]['seconds'] for k in ks),
               speedup=statistics.mean(T['full'][k]['seconds'] for k in ks) / statistics.mean(T[m][k]['seconds'] for k in ks),
               full_calls=T[m][ks[0]]['full_calls'], latent_mse=statistics.mean(T[m][k]['latent_mse_to_full'] for k in ks))
    sk = [k for k in ks if (*k, m) in sc]
    if sk:
        row['n_scored'] = len(sk)
        row['clip'] = statistics.mean(sc[(*k, m)]['clip_score'] for k in sk)
        row['clip_minus_full'] = statistics.mean(sc[(*k, m)]['clip_score'] - sc[(*k, 'full')]['clip_score'] for k in sk if (*k, 'full') in sc)
        ps = [sc[(*k, m)]['psnr_to_full'] for k in sk if sc[(*k, m)]['psnr_to_full'] is not None]
        row['psnr'] = statistics.mean(ps) if ps else None
        row['tde'] = statistics.mean(sc[(*k, m)]['temporal_delta_error'] for k in sk)
    summary['methods'][m] = row

comparisons = {}
for b in ['B1', 'B2', 'B3']:
    for a, base in [('ridge_dagger', 'ridge_off2x'), ('ridge_dagger', 'ridge_off'), ('ridge_off', 'zero'), ('ridge_dagger', 'zero'),
                    ('ridge_dagger2', 'ridge_off2x'), ('zero', 'linear')]:
        A, B = f'{b}_{a}', f'{b}_{base}'
        if A not in T or B not in T:
            continue
        for metric, src in [('latent_mse', 'ev'), ('psnr', 'sc'), ('clip', 'sc')]:
            d = defaultdict(list)
            for k in keys:
                if src == 'ev':
                    if k in T[A] and k in T[B]:
                        x, y = T[A][k]['latent_mse_to_full'], T[B][k]['latent_mse_to_full']
                        d[k[0]].append((y - x) / y)  # relative reduction; >0 means A better
                else:
                    ra, rb = sc.get((*k, A)), sc.get((*k, B))
                    if ra and rb and ra.get('psnr_to_full') is not None:
                        d[k[0]].append(ra['psnr_to_full'] - rb['psnr_to_full'] if metric == 'psnr' else ra['clip_score'] - rb['clip_score'])
            if len(d) >= 4:
                allv = [v for vs in d.values() for v in vs]
                lo, hi = boot(d)
                comparisons[f'{A} vs {B} [{metric}]'] = dict(mean=statistics.mean(allv), ci95=[lo, hi], win_rate=sum(v > 0 for v in allv) / len(allv), n=len(allv))
summary['comparisons'] = comparisons
(root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(f"{'method':28s} {'n':>3s} {'sec':>6s} {'spd':>5s} {'calls':>5s} {'lat_mse':>8s} {'psnr':>6s} {'clip':>6s} {'dclip':>6s}")
for m, r in summary['methods'].items():
    print(f"{m:28s} {r['n']:3d} {r['seconds']:6.2f} {r['speedup']:5.2f} {r['full_calls']:5d} {r['latent_mse']:8.4f} "
          f"{(r.get('psnr') or 0):6.2f} {(r.get('clip') or 0):6.2f} {(r.get('clip_minus_full') or 0):6.2f}")
for k, v in comparisons.items():
    print(f"{k:55s} mean={v['mean']:+.4f} ci=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] win={v['win_rate']:.2f} n={v['n']}")
