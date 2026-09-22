#!/usr/bin/env python3
"""Diagnostic matched-frame-difference postfilter sweep, NOT held-out tuning."""
import json
from pathlib import Path
import torch
from tis_review import spectral, metrics

ROOT=Path('/root/viewdit/results/video/tis_review')
torch.set_num_threads(2)
rows=[]
for seed in range(2):
    data=torch.load(ROOT/f'seed{seed}_videos.pt',map_location='cpu')
    ref=data['videos'][0]
    candidates=[]
    for cut in [4,8,12,15]:
        for retention in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            x=spectral(ref.unsqueeze(0),retention,cut)[0].clamp(-1,1)
            candidates.append((f'pixel_dct_cut{cut}_retain{retention}',metrics(x,ref)))
    for passes in [1,2,4,8,16]:
        for lam in [0.25,0.5,0.75,1.0]:
            x=ref.clone()
            for _ in range(passes):
                y=x.clone()
                y[1:-1]=(1-lam/2)*x[1:-1]+lam/4*(x[:-2]+x[2:])
                x=y
            candidates.append((f'pixel_smooth_pass{passes}_lam{lam}',metrics(x,ref)))
    for target in ['dct_weak','dct_zero','smooth']:
        m=metrics(data['videos'][data['names'].index(target)],ref)
        eligible=[(name,q) for name,q in candidates if q['frame_diff_proxy']<=m['frame_diff_proxy'] and q['spatial_gradient']>=m['spatial_gradient'] and q['psnr_vs_full']>=m['psnr_vs_full']]
        best=max(eligible,key=lambda v:v[1]['psnr_vs_full']) if eligible else None
        row={'seed':seed,'target':target,'target_metrics':m,'dominated_on_three_proxies':bool(best),'best_postfilter':best}
        rows.append(row)
        print(json.dumps(row),flush=True)
(ROOT/'postfilter_audit.json').write_text(json.dumps({'caveat':'Per-sample oracle diagnostic; three proxies are not sufficient to prove perceptual superiority. Not a held-out baseline comparison.','rows':rows},indent=2))
