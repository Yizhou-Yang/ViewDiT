#!/usr/bin/env python3
"""Exact error floor for the FIXED-prefix, future-latent-only control class."""
import json
from pathlib import Path
import torch
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_phase_audit')


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(ROOT.glob('*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=True);v=d['videos']
        base=v['base'];edited=v['edit_latent_only'];fixed=v['tail_compensated']
        assert torch.equal(edited[:,:9],fixed[:,:9])
        for end in [6,7,8,9]:
            frozen_residual=(edited[:,end:9]-base[:,end:9]).square().sum()
            denominator=base[:,end:].numel()
            floor=float(frozen_residual/denominator)
            initial=float((edited[:,end:]-base[:,end:]).square().mean())
            achieved=float((fixed[:,end:]-base[:,end:]).square().mean())
            rows.append({'sample':path.stem,'protected_suffix_start':end,'control_latent_start':3,'first_controlled_pixel_frame':9,'fixed_prefix_error_floor_mse':floor,'initial_suffix_mse':initial,'compensated_suffix_mse':achieved,'unavoidable_fraction_of_initial_mse_in_this_control_class':floor/initial,'note':'Bound follows from exact invariance of output before frame9 when only latent indices>=3 change. Does not apply to within-block channel controls or decoder modifications.'})
    result={'complete':True,'protocol':{'frames_zero_based':True,'end_offsets':[6,7,8,9],'restricted_control':'freeze latent 0:3, modify latent3 onward','proof':'causal decoding makes frames0:9 independent of latent3 onward; sum squared residual on protected frames below9 is invariant'},'rows':rows}
    (OUT/'stats.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
