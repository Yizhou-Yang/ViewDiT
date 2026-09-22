#!/usr/bin/env python3
"""Hard-preserved endpoint residual feedback, a feasibility prototype.
No semantic-quality claims. Synthetic native-edit localization targets.
"""
import os
import json
import time
from pathlib import Path
import torch
import endpoint_control as ec
from run_xl_gate import load_model, _forward, one_step, x0_from_eps
from tis_review import decode, metrics
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image
from PIL import Image

OUT=Path(os.environ.get('REFINE_OUT','/root/viewdit/results/video/endpoint_refine'))
N=int(os.environ.get('CONTROL_N','4'))
SEED0=int(os.environ.get('CONTROL_SEED0','40'))
ITERS=8
RATE=0.8


@torch.no_grad()
def run(model,diff,states,rng,delta):
    torch.cuda.set_rng_state(rng)
    x=states[0]+delta
    wrapped=diff._wrap_model(model)
    for j,idx in enumerate(reversed(range(diff.num_timesteps-ec.START))):
        t=torch.full((1,),idx,device=x.device,dtype=torch.long)
        x=one_step(diff,_forward(wrapped,x,t),x,t)
        x[:,:6]=states[j+1][:,:6];x[:,10:]=states[j+1][:,10:]
    return x.float()


def restricted(d):
    y=d.clone();y[:,:6]=0;y[:,10:]=0
    return y


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    model,diff=load_model(torch.device('cuda'))
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema',local_files_only=True).to('cuda').eval()
    report={'protocol':{'seeds':list(range(SEED0,SEED0+N)),'iterations':ITERS,'rate':RATE,'start':ec.START,'task':'synthetic model-native edit localization, not natural-language editing','amplitude':0.2,'selection':'fixed eight updates, final iterate; no best-sample selection'},'rows':[],'complete':False}
    def save():
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in range(SEED0,SEED0+N):
        tic=time.time()
        states,rng=ec.base_run(model,diff,seed)
        base=states[-1].float()
        field=ec.field_of(base)
        gu=torch.ones(16,device='cuda')*0.2
        global_y=ec.rollout(model,diff,states,rng,field,gu)
        env=torch.zeros(16,device='cuda');env[6:10]=torch.tensor([0.5,1,1,0.5],device='cuda')
        target=base+(global_y-base)*env.reshape(1,16,1,1,1)
        initial=field*(gu*env).reshape(1,16,1,1,1)
        vids={'base':decode(vae,base),'global_edit':decode(vae,global_y),'target_composite':decode(vae,target)}
        def record(name,y,extra=None):
            img=decode(vae,y)
            row=ec.score(y,base,target)
            row.update(seed=seed,method=name,pixel=metrics(img,vids['base']),pixel_target_mse=float((img[6:10]-vids['target_composite'][6:10]).square().mean()),pixel_outside_max_error=float((torch.cat([img[:6],img[10:]])-torch.cat([vids['base'][:6],vids['base'][10:]])).abs().max()))
            if extra: row.update(extra)
            report['rows'].append(row);vids[name]=img
            print('ROW',json.dumps(row),flush=True);save()
        y0=run(model,diff,states,rng,initial)
        record('hard_direct',y0)
        t=torch.full((1,),diff.num_timesteps-ec.START-1,device='cuda',dtype=torch.long)
        alpha=torch.as_tensor(diff.alphas_cumprod[t.item()],device='cuda').float().reshape(1,1,1,1,1)
        wrapped=diff._wrap_model(model)
        p0=x0_from_eps(states[0],_forward(wrapped,states[0],t)[:,:,:4],alpha).float()
        delta=initial.clone()
        for k in range(ITERS):
            x=states[0]+delta
            p=x0_from_eps(x,_forward(wrapped,x,t)[:,:,:4],alpha).float()
            delta=restricted(delta+RATE*((target-base)-(p-p0)))
        record('single_step_feedback',run(model,diff,states,rng,delta),{'extra_forward':ITERS})
        delta=initial.clone();y=y0
        errors=[ec.score(y,base,target)['target_rel_error']]
        start=time.time()
        for k in range(ITERS):
            delta=restricted(delta+RATE*(target-y))
            y=run(model,diff,states,rng,delta)
            errors.append(ec.score(y,base,target)['target_rel_error'])
        record('endpoint_feedback',y,{'error_curve':errors,'feedback_seconds':time.time()-start,'extra_forward':ITERS*(diff.num_timesteps-ec.START)})
        report.setdefault('clips',[]).append({'seed':seed,'seconds':time.time()-tic})
        save_image(make_grid(torch.cat([(v[[4,6,8,10]]+1)/2 for v in vids.values()]),nrow=4),OUT/f'seed{seed}.png')
        torch.save({'videos':vids},OUT/f'seed{seed}.pt')
        frames=[]
        keys=['base','hard_direct','single_step_feedback','endpoint_feedback','target_composite']
        for f in range(16):
            strip=torch.cat([(vids[k][f]+1)/2 for k in keys],dim=2)
            arr=(strip.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            frames.append(Image.fromarray(arr))
        frames[0].save(OUT/f'seed{seed}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
        save()
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
