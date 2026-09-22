#!/usr/bin/env python3
"""Equal-iteration full latent fitting baseline and latent reuse stress test."""
import json
import time
from pathlib import Path
import torch
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix

ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_baseline_audit')


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'iterations':24,'lr':.03,'baseline':'full-latent Adam, equal iterations but not equal runtime; normalized core and suffix losses','second_operation':'same small fresh geometric perturbation to last latent, test exact prefix retention; not a DiT semantic re-edit','warm_start':'suffix optimizer starts from hybrid reencoded splice','caveat':'The targets are still compositing targets; this evaluates latent representation fidelity, not superiority to rendered compositing.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        (OUT/'stats.json').write_text(json.dumps(report,indent=2))
    for path in sorted(ROOT.glob('*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=True)
        ze=d['edited_z'].cuda();zc=d['compensated_z'].cuda()
        base=d['videos']['base'].cuda();edited=d['videos']['edit_latent_only'].cuda();target=d['videos']['pixel_splice'].cuda()
        with torch.no_grad():
            zr=vae.model.encode(target.unsqueeze(0),vae.scale)[0].float()
            origerr=(edited[:,9:]-base[:,9:]).square().mean().clamp(min=1e-12)
            coreden=(edited[:,5:9]-base[:,5:9]).square().mean().clamp(min=1e-12)
            _,states=stream(vae,ze);boundary=states[3]
        tic=time.time();delta=torch.zeros_like(ze,requires_grad=True);opt=torch.optim.Adam([delta],lr=.03)
        torch.cuda.reset_peak_memory_stats()
        for k in range(24):
            opt.zero_grad(set_to_none=True)
            out=vae.model.decode((ze+delta).unsqueeze(0),vae.scale)[0].float().clamp(-1,1)
            loss=(out[:,9:]-target[:,9:]).square().mean()/origerr+(out[:,:9]-target[:,:9]).square().mean()/coreden+.001*delta.square().mean()/(ze.square().mean()+1e-8)
            assert torch.isfinite(loss)
            loss.backward();opt.step()
        with torch.no_grad():
            full,_=stream(vae,ze+delta)
            save({'sample':path.stem,'method':'full_latent_Adam','suffix_mse':float((full[:,9:]-base[:,9:]).square().mean()),'prefix_core_mse':float((full[:,:9]-edited[:,:9]).square().mean()),'prefix_core_max':float((full[:,:9]-edited[:,:9]).abs().max()),'seconds':time.time()-tic,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        tic=time.time();correction=(zr[:,3:]-ze[:,3:]).detach().requires_grad_(True);opt=torch.optim.Adam([correction],lr=.03)
        for k in range(24):
            opt.zero_grad(set_to_none=True)
            out=suffix(vae,ze[:,3:]+correction,boundary)
            loss=(out-base[:,9:]).square().mean()/origerr+.001*correction.square().mean()/(ze[:,3:].square().mean()+1e-8)
            loss.backward();opt.step()
        with torch.no_grad():
            zw=ze.clone();zw[:,3:]+=correction
            warm,_=stream(vae,zw)
            save({'sample':path.stem,'method':'warm_start_suffix','suffix_mse':float((warm[:,9:]-base[:,9:]).square().mean()),'prefix_core_max':float((warm[:,:9]-edited[:,:9]).abs().max()),'seconds':time.time()-tic})
            field=(torch.roll(ze[:,-1],1,-1)-torch.roll(ze[:,-1],-1,-1))*.025
            for name,z in [('compensated',zc),('hybrid_reencode',torch.cat([ze[:,:3],zr[:,3:]],1)),('full_reencode',zr)]:
                zz=z.clone();zz[:,-1]+=field
                output,_=stream(vae,zz)
                prior,_=stream(vae,z)
                save({'sample':path.stem,'method':'reuse_'+name,'prefix_core_vs_edit_max':float((output[:,:9]-edited[:,:9]).abs().max()),'preceding_frames_vs_prior_max':float((output[:,:13]-prior[:,:13]).abs().max()),'suffix_mse_vs_original':float((output[:,9:]-base[:,9:]).square().mean()),'last_chunk_change_rms':float((output[:,13:]-prior[:,13:]).square().mean().sqrt())})
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
