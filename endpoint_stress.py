#!/usr/bin/env python3
"""Audit solver attribution and test trajectory compositing on unconstrained bridges."""
import json
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers.models import AutoencoderKL
import endpoint_control as ec
from endpoint_validate import rollout
from run_xl_gate import load_model, _forward, one_step
from tis_review import decode

OUT = Path('/root/viewdit/results/video/endpoint_stress')

@torch.no_grad()
def native_path(model, diff, states, rng, delta):
    torch.cuda.set_rng_state(rng)
    x = states[0] + delta
    path = [x.clone()]
    wrapped = diff._wrap_model(model)
    for idx in reversed(range(diff.num_timesteps - ec.START)):
        t = torch.full((1,), idx, device='cuda', dtype=torch.long)
        x = one_step(diff, _forward(wrapped, x, t), x, t)
        path.append(x.clone())
    return path

@torch.no_grad()
def bridge_path(model, diff, states, native, rng, env, pin_core):
    torch.cuda.set_rng_state(rng)
    w = env.view(1,16,1,1,1)
    x = states[0] + w * (native[0] - states[0])
    wrapped = diff._wrap_model(model)
    for j, idx in enumerate(reversed(range(diff.num_timesteps - ec.START))):
        t = torch.full((1,), idx, device='cuda', dtype=torch.long)
        x = one_step(diff, _forward(wrapped, x, t), x, t)
        x[:,:4] = states[j+1][:,:4]
        x[:,12:] = states[j+1][:,12:]
        if pin_core:
            x[:,6:10] = native[j+1][:,6:10]
    return x.float()

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    model, diff = load_model(torch.device('cuda'))
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema', local_files_only=True).to('cuda').eval()
    report = {'complete':False,'protocol':{'ablation_seeds':[50,51,52,53], 'bridge_seeds':[60,61,62,63], 'budget':8, 'tail_steps':75, 'bridge_partition':{'P':[0,1,2,3,12,13,14,15], 'B':[4,5,10,11], 'E':[6,7,8,9]}, 'warning':'Geometry prototype; temporal smoothness is not quality; no target supplied to bridge frames.'}, 'rows':[]}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW',json.dumps(row),flush=True)
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in [50,51,52,53]:
        states,rng=ec.base_run(model,diff,seed)
        base=states[-1].float(); field=ec.field_of(base)
        env=torch.zeros(16,device='cuda')
        if seed%2==0:
            env[4:12]=torch.tensor([.25,.5,.75,1,1,.75,.5,.25],device='cuda')
        else:
            env[3:6]=torch.tensor([.5,1,.5],device='cuda');env[10:13]=torch.tensor([.5,1,.5],device='cuda')
        active=env>0; mask=active.view(1,16,1,1,1)
        native=ec.rollout(model,diff,states,rng,field,torch.full((16,),.2,device='cuda'))
        target=base+(native-base)*env.view(1,16,1,1,1)
        delta=.2*field*env.view(1,16,1,1,1)
        y=rollout(model,diff,states,rng,delta,active)
        denom=(target-base).norm()
        error=lambda a:float(((target-a)*mask).norm()/denom)
        errors=[error(y)]; trials=[];eta=.8; tic=time.time()
        for k in range(8):
            candidate_delta=(delta+eta*(target-y))*mask
            candidate=rollout(model,diff,states,rng,candidate_delta,active)
            accept=bool(torch.isfinite(candidate).all()) and error(candidate)<error(y)
            trials.append({'error':error(candidate),'accepted':accept,'eta':eta})
            if accept: delta,y=candidate_delta,candidate
            else: eta=max(.05,eta*.5)
            errors.append(error(y))
        img=decode(vae,y); ti=decode(vae,target)
        save({'task':'ablation','seed':seed,'method':'reject_only','target_rel_error':error(y),'error_curve':errors,'trials':trials,'extra_nfe':600,'seconds':time.time()-tic,'pixel_target_mse':float((img[active.cpu()]-ti[active.cpu()]).square().mean())})
    for seed in [60,61,62,63]:
        states,rng=ec.base_run(model,diff,seed)
        base=states[-1].float();field=ec.field_of(base)
        native=native_path(model,diff,states,rng,.4*field)
        target=native[-1].float()
        env=torch.zeros(16,device='cuda');env[4:12]=torch.tensor([1/3,2/3,1,1,1,1,2/3,1/3],device='cuda')
        e=torch.zeros(16,device='cuda',dtype=torch.bool);e[6:10]=True
        active=torch.zeros_like(e);active[4:12]=True
        mask=e.view(1,16,1,1,1)
        denom=(target[:,e]-base[:,e]).norm()
        err=lambda y:float((y[:,e]-target[:,e]).norm()/denom)
        base_img=decode(vae,base);native_img=decode(vae,target)
        videos={'base':base_img,'native_edit':native_img}
        edge=torch.tensor([3,4,5,9,10,11])
        def record(name,y=None,img=None,extra=None):
            if img is None: img=decode(vae,y)
            d=img-base_img
            row={'task':'bridge','seed':seed,'method':name,'core_latent_rel_error':err(y) if y is not None else None,'core_pixel_mse':float((img[6:10]-native_img[6:10]).square().mean()),'preserve_pixel_max':float(torch.cat([d[:4],d[12:]]).abs().max()),'edit_residual_boundary_jump_proxy':float((d[1:]-d[:-1])[edge].square().mean().sqrt()),'edit_residual_acceleration_proxy':float((d[2:]-2*d[1:-1]+d[:-2]).square().mean().sqrt())}
            if extra:row.update(extra)
            videos[name]=img;save(row)
        hard=base.clone();hard[:,6:10]=target[:,6:10]
        record('hard_composite',hard)
        mixed=base+(target-base)*env.view(1,16,1,1,1)
        record('latent_crossfade',mixed)
        pix=base_img+(native_img-base_img)*env.cpu().view(16,1,1,1)
        record('pixel_crossfade',img=pix)
        delta=.4*field*env.view(1,16,1,1,1)
        y=rollout(model,diff,states,rng,delta,active)
        record('free_bridge_direct',y)
        errors=[err(y)];eta=.8;hist=[(delta.clone(),(target-y)*mask)];trials=[];tic=time.time()
        for k in range(8):
            r=(target-y)*mask; proposal=delta+eta*r
            if len(hist)>=2:
                recent=hist[-3:]; rs=torch.stack([h[1].flatten() for h in recent],dim=1).double()
                g=rs.T@rs;g+=torch.eye(len(recent),device='cuda')*(.001*g.diag().mean().clamp(min=1e-10))
                c=torch.linalg.solve(g,torch.ones(len(recent),device='cuda',dtype=torch.float64))
                if abs(float(c.sum()))>1e-10:
                    c=c/c.sum()
                    if float(c.abs().sum())<=4:proposal=sum(float(w)*(d+eta*r) for w,(d,r) in zip(c,recent))
            proposal=proposal*active.view(1,16,1,1,1)
            cand=rollout(model,diff,states,rng,proposal,active)
            accept=bool(torch.isfinite(cand).all()) and err(cand)<err(y)
            trials.append({'error':err(cand),'accepted':accept,'eta':eta})
            if accept:
                delta,y=proposal,cand;hist.append((delta.clone(),(target-y)*mask));hist=hist[-3:]
            else:eta=max(.05,eta*.5);hist=[(delta.clone(),(target-y)*mask)]
            errors.append(err(y))
        record('core_feedback_free_bridge',y,extra={'extra_nfe':600,'seconds':time.time()-tic,'error_curve':errors,'trials':trials})
        y=bridge_path(model,diff,states,native,rng,env,True)
        record('trajectory_pins_free_bridge',y,extra={'extra_nfe_vs_direct':0,'note':'Standard trajectory inpainting baseline; final core pin makes core exact, not a novel method.'})
        torch.save({'videos':videos},OUT/f'seed{seed}.pt')
        keys=['base','latent_crossfade','core_feedback_free_bridge','trajectory_pins_free_bridge','native_edit']
        frames=[]
        for f in range(16):
            strip=torch.cat([(videos[k][f]+1)/2 for k in keys],dim=2)
            arr=(strip.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            frame=Image.new('RGB',(1280,284),'white');frame.paste(Image.fromarray(arr),(0,28));draw=ImageDraw.Draw(frame)
            for j,k in enumerate(keys):draw.text((j*256+4,6),k,fill='black')
            frames.append(frame)
        frames[0].save(OUT/f'seed{seed}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
        panel=Image.new('RGB',(1280,1136),'white')
        for j,f in enumerate([4,5,10,11]):panel.paste(frames[f],(0,j*284))
        panel.save(OUT/f'seed{seed}.png')
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
