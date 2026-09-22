#!/usr/bin/env python3
"""Localize a model-native edit instead of imposing an arbitrary tensor warp.
Calibration: 16 terminal response probes. A global native edit supplies the
appearance transformation; requested temporal envelope is held out of probes.
Includes single-step PMP, diagonal, full response and hard preservation.
"""
import os
import json
import time
from pathlib import Path
import torch
import endpoint_control as ec
from run_xl_gate import load_model, _forward, x0_from_eps
from tis_review import decode, metrics
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image

OUT=Path(os.environ.get('NATIVE_OUT','/root/viewdit/results/video/endpoint_native'))
N=int(os.environ.get('CONTROL_N','2'))
SEED0=int(os.environ.get('CONTROL_SEED0','30'))


def solve(J,want,weights,inside_only=False):
    a=J.double()*weights.reshape(-1,1).double()
    b=want.reshape(-1).double()*weights.reshape(-1).double()
    gram=a.T@a
    ridge=0.005*gram.diag().mean().clamp(min=1e-10)
    return torch.linalg.solve(gram+ridge*torch.eye(16,device=a.device,dtype=a.dtype),a.T@b).float().clamp(-0.8,0.8)


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    model,diff=load_model(torch.device('cuda'))
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema',local_files_only=True).to('cuda').eval()
    wrapped=diff._wrap_model(model)
    report={'protocol':{'seeds':list(range(SEED0,SEED0+N)),'start':ec.START,'h':ec.H,'task':'localize model-native edit; synthetic fixed spatial direction; no text semantics','full_response_probes':16,'single_step_probes':16,'target':'masked global model-native endpoint change; not arbitrary warped latent'},'rows':[],'complete':False}
    def save():
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in range(SEED0,SEED0+N):
        tic=time.time()
        states,rng=ec.base_run(model,diff,seed)
        base=states[-1].float()
        field=ec.field_of(base)
        zero=torch.zeros(16,device='cuda')
        replay=ec.rollout(model,diff,states,rng,field,zero)
        assert torch.equal(replay,base)
        t=torch.full((1,),diff.num_timesteps-ec.START-1,device='cuda',dtype=torch.long)
        alpha=torch.as_tensor(diff.alphas_cumprod[t.item()],device='cuda',dtype=torch.float32).reshape(1,1,1,1,1)
        out=_forward(wrapped,states[0],t)
        pmp=x0_from_eps(states[0],out[:,:,:4],alpha).float()
        cols=[];pc=[]
        for f in range(16):
            u=zero.clone();u[f]=ec.H
            y=ec.rollout(model,diff,states,rng,field,u)
            cols.append(((y-base)/ec.H).flatten())
            xp=states[0]+field*u.reshape(1,16,1,1,1)
            op=_forward(wrapped,xp,t)
            pp=x0_from_eps(xp,op[:,:,:4],alpha).float()
            pc.append(((pp-pmp)/ec.H).flatten())
        J=torch.stack(cols,1);P=torch.stack(pc,1)
        JD=J.reshape(1,16,4,32,32,16).clone()
        for f in range(16):
            JD[:,f]*=torch.eye(16,device='cuda')[f]
        JD=JD.reshape(-1,16)
        weights=torch.ones_like(base);weights[:,:6]=2;weights[:,10:]=2
        vids={'base':decode(vae,base)}
        for amp in [0.1,0.2]:
            global_u=torch.ones(16,device='cuda')*amp
            global_y=ec.rollout(model,diff,states,rng,field,global_u)
            envelope=zero.clone();envelope[6:10]=torch.tensor([0.5,1,1,0.5],device='cuda')
            want=(global_y-base)*envelope.reshape(1,16,1,1,1)
            target=base+want
            direct_u=global_u*envelope
            full_u=solve(J,want,weights)
            diag_u=solve(JD,want,weights)
            pmp_u=solve(P,want,weights)
            methods=[('direct',direct_u,False),('single_step',pmp_u,False),('diagonal',diag_u,False),('response',full_u,False),('hard_preserve',direct_u,True)]
            for name,u,clamp in methods:
                y=ec.rollout(model,diff,states,rng,field,u,clamp)
                row=ec.score(y,base,target)
                row.update(seed=seed,amp=amp,method=name,coefficient_norm=float(u.norm()),prediction_rel_error=float(((y-base)-(J@u).reshape_as(base)).norm()/((y-base).norm()+1e-8)))
                img=decode(vae,y)
                refimg=vids['base']
                targetimg=decode(vae,target)
                row['pixel']=metrics(img,refimg)
                row['pixel_target_mse']=float((img[6:10]-targetimg[6:10]).square().mean())
                row['pixel_outside_mse']=float((torch.cat([img[:6],img[10:]])-torch.cat([refimg[:6],refimg[10:]])).square().mean())
                report['rows'].append(row)
                print('ROW',json.dumps(row),flush=True)
                if amp==0.2:
                    vids[name]=img
                save()
            if amp==0.2:
                vids['global_edit']=decode(vae,global_y);vids['target_composite']=decode(vae,target)
        report.setdefault('clips',[]).append({'seed':seed,'seconds':time.time()-tic,'offdiag_energy':float((J-JD).square().sum()/J.square().sum())})
        save_image(make_grid(torch.cat([(v[[4,6,8,10]]+1)/2 for v in vids.values()]),nrow=4),OUT/f'seed{seed}.png')
        torch.save({'videos':vids,'J':J.cpu(),'P':P.cpu()},OUT/f'seed{seed}.pt')
        save()
    report['complete']=True
    save()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
