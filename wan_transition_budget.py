#!/usr/bin/env python3
"""Hard core preservation with explicit minimum tested transition budget."""
import json
import time
from pathlib import Path
import torch
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_transition_budget')

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'core_end':7,'protected_starts':[7,9,11,13],'iterations':24,'lr':.03,'input_core':'freeze latent indices0,1,2, guarantees frames0..8 unchanged','feasibility_threshold':'suffix RMSE <= 0.1 * core edit RMS; NOT statistical significance','selection':'smallest TESTED transition budget meeting threshold; not global minimum','free_bridge':'No target supplied to bridge. Original masked future latents optimized only for protected suffix. Smoothness diagnostic not objective.'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for p in sorted(ROOT.glob('*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True);z=d['edited_z'].cuda();base=d['videos']['base'].cuda();edit=d['videos']['edit_latent_only'].cuda()
        with torch.no_grad():
            _,states=stream(vae,z);state=states[3];core=(edit[:,5:7]-base[:,5:7]).square().mean().sqrt()
        for start in [7,9,11,13]:
            delta=torch.zeros_like(z[:,3:],requires_grad=True);opt=torch.optim.Adam([delta],lr=.03);tic=time.time()
            initial=(edit[:,start:]-base[:,start:]).square().mean().clamp(min=1e-12)
            for step in range(24):
                opt.zero_grad(set_to_none=True);y=torch.cat([edit[:,:9],suffix(vae,z[:,3:]+delta,state)],1)
                loss=(y[:,start:]-base[:,start:]).square().mean()/initial+.001*delta.square().mean()
                loss.backward();opt.step()
            with torch.no_grad():
                zz=z.clone();zz[:,3:]+=delta;final,_=stream(vae,zz)
                alpha=torch.ones(17,device='cuda');alpha[start:]=0
                if start>7:alpha[7:start]=torch.linspace(1,0,start-7+2,device='cuda')[1:-1]
                pixel_target=base+alpha.view(1,17,1,1)*(edit-base)
                rz=vae.model.encode(pixel_target.unsqueeze(0),vae.scale)[0].float();ry,_=stream(vae,rz)
                rms=(final[:,start:]-base[:,start:]).square().mean().sqrt()
                r={'sample':p.stem,'protected_start':start,'transition_frames':start-7,'core_rms':float(core),'core_max_error':float((final[:,:7]-edit[:,:7]).abs().max()),'protected_rms':float(rms),'relative_protected_rms':float(rms/core),'initial_relative_protected_rms':float(initial.sqrt()/core),'passes_budget':bool(rms/core<=.1),'reencoded_fade_core_relative_rmse':float((ry[:,5:7]-edit[:,5:7]).square().mean().sqrt()/core),'reencoded_fade_protected_relative_rms':float((ry[:,start:]-base[:,start:]).square().mean().sqrt()/core),'seconds':time.time()-tic,'residual_accel_rms':float(((final-base)[:,2:]-2*(final-base)[:,1:-1]+(final-base)[:,:-2]).square().mean().sqrt())}
                save(r)
                torch.save({'latent':zz.cpu(),'videos':{'base':base.cpu(),'edit':edit.cpu(),'free_bridge':final.cpu(),'reencoded_fade':ry.cpu(),'pixel_fade':pixel_target.cpu()}},OUT/f'{p.stem}_start{start}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
