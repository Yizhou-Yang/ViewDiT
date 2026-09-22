#!/usr/bin/env python3
"""Finite-difference cross-frame coupling, not a new method claim.
Paired symmetric perturbations; no backward pass; two amplitude check.
"""
import json
import os
from pathlib import Path
import torch
from run_xl_gate import load_model, _noise, _forward, one_step, x0_from_eps
from tis_probe import frame_lowpass

OUT = Path(os.environ.get('VIEWDIT_VIDEO_OUT','/root/viewdit/results/video/tis_coupling'))


def adjacent(x):
    y=x[0].float().flatten(1)
    return float(torch.nn.functional.cosine_similarity(y[:-1],y[1:],dim=1).mean())


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.set_grad_enabled(False)
    device=torch.device('cuda')
    test=torch.randn(1,16,4,8,8,device=device)
    correct=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(test.permute(0,2,3,4,1).reshape(-1,1,16),(1,1),mode='replicate'),3,stride=1).reshape(1,4,8,8,16).permute(0,4,1,2,3)
    report={'wrong_axis_relative_error':float((frame_lowpass(test,3)-correct).norm()/correct.norm()),'rows':[],'complete':False}
    model,diffusion=load_model(device)
    wrapped=diffusion._wrap_model(model)
    alphas=torch.as_tensor(diffusion.alphas_cumprod,device=device,dtype=torch.float32)
    for seed in range(2):
        torch.manual_seed(9000+seed)
        torch.cuda.manual_seed_all(9000+seed)
        x=_noise(device,1000+seed)
        gen=torch.Generator(device=device).manual_seed(7654+seed)
        direction=torch.randn(x[:,7].shape,device=device,generator=gen)
        direction=direction/direction.square().mean().sqrt()
        for i,idx in enumerate(reversed(range(diffusion.num_timesteps))):
            t=torch.full((1,),idx,device=device,dtype=torch.long)
            out=_forward(wrapped,x,t)
            if i in [25,125,225]:
                derivatives=[]
                for eta in [0.02,0.04]:
                    delta=torch.zeros_like(x)
                    delta[:,7]=eta*x.float().square().mean().sqrt()*direction
                    ep=_forward(wrapped,x+delta,t)[:,:,:4].float()
                    em=_forward(wrapped,x-delta,t)[:,:,:4].float()
                    derivatives.append((ep-em)/(2*eta))
                de=derivatives[0]
                energy=de.square().sum(dim=(0,2,3,4))
                other=torch.arange(16,device=device)!=7
                alpha=alphas[idx].view(1,1,1,1,1)
                report['rows'].append({'seed':seed,'iteration':i,'alpha_bar':float(alphas[idx]),'eps_adj_cos':adjacent(out[:,:,:4]),'x0_adj_cos':adjacent(x0_from_eps(x,out[:,:,:4],alpha)),'off_frame_response_energy_fraction':float(energy[other].sum()/energy.sum()),'off_frame_response_norm':float(de[:,other].norm()),'two_amplitude_relative_disagreement':float((derivatives[0]-derivatives[1]).norm()/(derivatives[1].norm()+1e-12))})
                print(json.dumps(report['rows'][-1]),flush=True)
            x=one_step(diffusion,out,x,t)
    report['complete']=True
    (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
