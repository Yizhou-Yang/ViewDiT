#!/usr/bin/env python3
"""Monotonicity diagnostic: is terminal fidelity monotone in the fraction of cache error removed?
dh_used = dh_cached + lam * (dh_true - dh_cached), same for text stream (mode oracle_both) or video only.
lam=0 is zero-order reuse, lam=1 recovers full compute exactly. Also measures a perturbation floor
(full compute from a slightly perturbed initial noise) to separate trajectory sensitivity from bias.
Oracle arms cost full compute; this is diagnosis, not an acceleration method."""
import json, sys
from pathlib import Path
import h20_cache_corrector as m

out = Path('/data/viewdit/results/diag/monotonicity.jsonl')
R = m.Runner(out.parent)
prompts = [('val0', m.VAL[0]), ('val1', m.VAL[1]), ('cal0', m.CAL[0]), ('cal3', m.CAL[3]), ('cal5', m.CAL[5]), ('cal6', m.CAL[6])]
done = set()
if out.exists():
    done = {(r['prompt'], r['budget'], r['mode'], r['lam']) for r in map(json.loads, open(out))}
for tag, p in prompts:
    ref, _ = R.run(p, 1)
    for bud in ['B2', 'B1']:
        b, k = m.BUDGETS[bud]
        arms = [('zero', None)] + [(md, l) for md in ['oracle_both', 'oracle_dh'] for l in [0.25, 0.5, 0.75, 0.9]] + [('perturb', 1e-3), ('perturb', 1e-2)]
        for mode, lam in arms:
            if (tag, bud, mode, lam) in done or (mode == 'perturb' and bud != 'B2'):
                continue
            if mode == 'perturb':
                z, _ = R.run(p, 1, perturb=lam)
            else:
                z, _ = R.run(p, 1, blocks=b, k=k, mode=mode, W=lam)
            r = dict(prompt=tag, budget=bud, mode=mode, lam=lam, latent_mse=float((z - ref).float().square().mean()))
            with open(out, 'a') as f:
                f.write(json.dumps(r) + '\n')
            print(json.dumps(r), flush=True)
