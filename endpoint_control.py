#!/usr/bin/env python3
"""Endpoint response precompensation: mechanism prototype, not semantic editing.
Run in existing ViewDiT experiments/video environment. All fitting uses model
rollouts, not unseen reference data. Target is a synthetic latent displacement.
This tests endpoint controllability only; pixel warp is an essential baseline.
"""
import os
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_xl_gate import load_model, _noise, _forward, one_step
from tis_review import decode, metrics
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image

OUT = Path(os.environ.get('CONTROL_OUT','/root/viewdit/results/video/endpoint_control'))
N = int(os.environ.get('CONTROL_N','2'))
START = int(os.environ.get('CONTROL_START','175'))
SEED0 = int(os.environ.get('CONTROL_SEED0','20'))
H = 0.12
RIDGE = 0.01


@torch.no_grad()
def base_run(model,diff,seed):
    torch.manual_seed(seed+9000)
    torch.cuda.manual_seed_all(seed+9000)
    x=_noise(torch.device('cuda'),seed+1000)
    wrapped=diff._wrap_model(model)
    states=[]
    rng=None
    for i,idx in enumerate(reversed(range(diff.num_timesteps))):
        if i==START:
            rng=torch.cuda.get_rng_state()
        if i>=START:
            states.append(x.clone())
        t=torch.full((1,),idx,device=x.device,dtype=torch.long)
        x=one_step(diff,_forward(wrapped,x,t),x,t)
    states.append(x.clone())
    return states,rng


@torch.no_grad()
def rollout(model,diff,states,rng,field,u,clamp=False):
    torch.cuda.set_rng_state(rng)
    x=states[0]+field*u.view(1,16,1,1,1)
    wrapped=diff._wrap_model(model)
    for j,idx in enumerate(reversed(range(diff.num_timesteps-START))):
        t=torch.full((1,),idx,device=x.device,dtype=torch.long)
        x=one_step(diff,_forward(wrapped,x,t),x,t)
        if clamp:
            x[:,:6]=states[j+1][:,:6]
            x[:,10:]=states[j+1][:,10:]
    return x.float()


def field_of(x):
    yy,xx=torch.meshgrid(torch.linspace(-1,1,32,device=x.device),torch.linspace(-1,1,32,device=x.device),indexing='ij')
    mask=torch.exp(-((xx/0.65)**2+((yy-0.3)/0.45)**2)/2)[None,None,None]
    delta=(torch.roll(x.float(),1,-1)-torch.roll(x.float(),-1,-1))/2*mask
    return delta/(delta.square().mean().sqrt()+1e-8)*x.float().square().mean().sqrt()


def score(y,base,target):
    d=y-base
    want=target-base
    inside=slice(6,10)
    outside=torch.tensor([0,1,2,3,4,5,10,11,12,13,14,15],device=y.device)
    target_norm=want[:,inside].norm().clamp(min=1e-8)
    return {'target_rel_error':float((y[:,inside]-target[:,inside]).norm()/target_norm),
            'outside_leak_rel':float(d[:,outside].norm()/target_norm),
            'full_target_rel_error':float((y-target).norm()/target_norm),
            'achieved_target_gain':float((d*want).sum()/(want.square().sum()+1e-8))}


def solve(J,want,weights):
    a=J.double()*weights.reshape(-1,1).double()
    b=want.reshape(-1).double()*weights.reshape(-1).double()
    gram=a.T@a
    ridge=RIDGE*gram.diag().mean().clamp(min=1e-10)
    return torch.linalg.solve(gram+ridge*torch.eye(16,device=a.device,dtype=a.dtype),a.T@b).float().clamp(-0.6,0.6)


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    model,diff=load_model(torch.device('cuda'))
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema',local_files_only=True).to('cuda').eval()
    report={'protocol':{'seeds':list(range(SEED0,SEED0+N)),'start':START,'steps':diff.num_timesteps,'finite_difference':H,'ridge':RIDGE,'task':'synthetic local latent displacement, NOT semantic edit or quality benchmark','fit':'full terminal latent, 16 one-sided common-noise probes per clip','max_coefficient':0.6},'rows':[],'complete':False}
    def save():
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in range(SEED0,SEED0+N):
        start=time.time()
        states,rng=base_run(model,diff,seed)
        base=states[-1].float()
        field=field_of(base)
        zero=torch.zeros(16,device='cuda')
        check=rollout(model,diff,states,rng,field,zero)
        check_error=float((check-base).abs().max())
        print('IDENTITY',seed,check_error,flush=True)
        assert check_error==0,'paired replay must match exactly'
        cols=[]
        for f in range(16):
            u=zero.clone();u[f]=H
            y=rollout(model,diff,states,rng,field,u)
            cols.append(((y-base)/H).flatten())
        J=torch.stack(cols,dim=1)
        weights=torch.ones_like(base);weights[:,:6]=2;weights[:,10:]=2
        JD=J.reshape(1,16,4,32,32,16).clone()
        for f in range(16):
            JD[:,f,:,:,:,:]*=torch.eye(16,device='cuda')[f]
        JD=JD.reshape(-1,16)
        images=[decode(vae,base)]
        names=['base']
        vid_out={'base':images[0]}
        for amp in [0.1,0.2]:
            target_u=zero.clone();target_u[6:10]=torch.tensor([0.5,1,1,0.5],device='cuda')*amp
            want=field*target_u.view(1,16,1,1,1)
            target=base+want
            u_full=solve(J,want,weights)
            u_diag=solve(JD,want,weights)
            candidates=[('direct',target_u,False),('diagonal',u_diag,False),('response',u_full,False),('hard_preserve',u_full,True)]
            for name,u,clamp in candidates:
                t0=time.time()
                y=rollout(model,diff,states,rng,field,u,clamp)
                row=score(y,base,target)
                prediction=(J@u).reshape_as(base)
                row.update(seed=seed,amp=amp,method=name,coefficient_norm=float(u.norm()),prediction_rel_error=float(((y-base)-prediction).norm()/((y-base).norm()+1e-8)),tail_seconds=time.time()-t0)
                img=decode(vae,y)
                row['pixel']=metrics(img,images[0])
                report['rows'].append(row)
                print('ROW',json.dumps(row),flush=True)
                if amp==0.2:
                    names.append(name);images.append(img);vid_out[name]=img
                save()
            if amp==0.2:
                target_img=decode(vae,target)
                names.append('synthetic_target');images.append(target_img);vid_out['synthetic_target']=target_img
        grid=make_grid(torch.cat([(v[[4,6,8,10]]+1)/2 for v in images]),nrow=4)
        save_image(grid,OUT/f'seed{seed}.png')
        torch.save({'videos':vid_out,'names':names,'J':J.cpu(),'field':field.cpu(),'base_latent':base.cpu()},OUT/f'seed{seed}.pt')
        report.setdefault('clips',[]).append({'seed':seed,'total_seconds':time.time()-start,'identity_max_error':check_error,'response_offdiag_energy':float((J-JD).square().sum()/J.square().sum())})
        save()
    report['complete']=True
    save()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
