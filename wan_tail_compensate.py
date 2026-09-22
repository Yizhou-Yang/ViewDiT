#!/usr/bin/env python3
"""Causal suffix compensation in a frozen Wan VAE; representation feasibility only."""
import json
import time
from pathlib import Path
import torch
from wan_vae_official import WanVAE
from wan_state_probe import stream, clone_cache

OUT=Path('/root/viewdit/results/video/wan_tail_compensate')


def suffix(vae,z,state):
    m=vae.model
    x=m.conv2(z.unsqueeze(0)/vae.scale[1].view(1,16,1,1,1)+vae.scale[0].view(1,16,1,1,1))
    cache=clone_cache(state);outputs=[]
    for i in range(x.shape[2]):
        outputs.append(m.decoder(x[:,:,i:i+1],feat_cache=cache,feat_idx=[0]))
    return torch.cat(outputs,2)[0].float().clamp(-1,1)


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'seeds':[40,41,42,43],'iterations':24,'lr':.03,'objective':'protected suffix pixel MSE normalized by initial MSE + 0.001 normalized latent correction MSE','trained_parameters':'none; only per-video future latent correction','fixed_output':'last iterate, no best-iterate selection','note':'same source clips as mechanism audit; not held-out semantic evaluation'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for seed in [40,41,42,43]:
        with torch.no_grad():
            z=torch.load(f'/root/viewdit/results/video/wan_state_probe/seed{seed}.pt',map_location='cpu',weights_only=True)['z'].cuda()
            base,_=stream(vae,z)
            field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5
            field=field/(field.square().mean().sqrt()+1e-8)*z.square().mean().sqrt()
            ze=z.clone();ze[:,2]+=.4*field[:,2]
            edited,states=stream(vae,ze)
            state=states[3];future=ze[:,3:].clone()
            y0=suffix(vae,future,state)
            identity=float((y0-edited[:,9:]).abs().max());assert identity==0
            target=base[:,9:];denom=(y0-target).square().mean().clamp(min=1e-12)
            splice=edited.clone();splice[:,9:]=base[:,9:]
            reencoded=vae.model.encode(splice.unsqueeze(0),vae.scale)[0]
            rerecon,_=stream(vae,reencoded)
            hybrid=ze.clone();hybrid[:,3:]=reencoded[:,3:]
            hybrid_recon,_=stream(vae,hybrid)
            report.setdefault('hybrid_baseline',[]).append({'seed':seed,'suffix_mse':float((hybrid_recon[:,9:]-target).square().mean()),'core_max_change':float((hybrid_recon[:,:9]-edited[:,:9]).abs().max())})
        correction=torch.zeros_like(future,requires_grad=True)
        optimizer=torch.optim.Adam([correction],lr=.03)
        tic=time.time();torch.cuda.reset_peak_memory_stats();curve=[]
        for k in range(24):
            optimizer.zero_grad(set_to_none=True)
            out=suffix(vae,future+correction,state)
            mse=(out-target).square().mean()
            loss=mse/denom+.001*correction.square().mean()/(future.square().mean()+1e-8)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            curve.append(float(mse.detach()/denom))
        with torch.no_grad():
            final=ze.clone();final[:,3:]=future+correction.detach()
            full,_=stream(vae,final)
            fast=suffix(vae,final[:,3:],state)
            assert float((full[:,9:]-fast).abs().max())==0
            row={'seed':seed,'initial_suffix_mse':float(denom),'final_suffix_mse':float((full[:,9:]-target).square().mean()),'suffix_mse_reduction_pct':float(100*(1-(full[:,9:]-target).square().mean()/denom)),'prefix_and_core_max_change':float((full[:,:9]-edited[:,:9]).abs().max()),'reencode_splice_suffix_mse':float((rerecon[:,9:]-target).square().mean()),'reencode_splice_core_mse':float((rerecon[:,:9]-edited[:,:9]).square().mean()),'relative_latent_correction':float(correction.norm()/future.norm()),'seconds':time.time()-tic,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'relative_mse_curve':curve,'suffix_replay_max':identity,'shape':list(z.shape)}
            save(row)
            torch.save({'videos':{'base':base.cpu(),'edited':edited.cpu(),'compensated':full.cpu(),'pixel_splice':splice.cpu(),'reencoded_splice':rerecon.cpu()},'compensated_z':final.cpu()},OUT/f'seed{seed}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
