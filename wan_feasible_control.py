#!/usr/bin/env python3
"""Feasible core preservation with block-selective backtracking; not a new optimizer."""
import json
import math
import time
from pathlib import Path
import torch
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_feasible_control')

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'core_relative_rmse_budget':.01,'steps':32,'lr':.03,'backtrack_scales':[1,.25,.0625,.015625,.00390625,0],'boundary':7,'warning':'Empirical feasible set under one optimizer, not an impossibility certificate. Block line search has extra forwards; report cost, do not call equal-time.'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for p in sorted(ROOT.glob('*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True);z=d['edited_z'].cuda();base=d['videos']['base'].cuda();edit=d['videos']['edit_latent_only'].cuda()
        with torch.no_grad():
            _,states=stream(vae,z);state=states[2];denom=(edit[:,7:]-base[:,7:]).square().mean();coreden=(edit[:,5:7]-base[:,5:7]).square().mean();bound=coreden*.0001
            floor=(edit[:,7:9]-base[:,7:9]).square().sum()/base[:,7:].numel()
        for method in ['freeze_boundary','global_backtrack','block_backtrack']:
            delta=torch.zeros_like(z[:,2:],requires_grad=True);opt=torch.optim.Adam([delta],lr=.03);tic=time.time();curve=[];forwards=0
            for step in range(32):
                opt.zero_grad(set_to_none=True);y=suffix(vae,z[:,2:]+delta,state);forwards+=1
                objective=(y[:,2:]-base[:,7:]).square().mean()/denom+.001*delta.square().mean()
                objective.backward()
                if method=='freeze_boundary':delta.grad[:,0]=0
                before=delta.detach().clone();opt.step()
                with torch.no_grad():
                    proposed=delta.detach().clone();chosen=None
                    scales=[0.] if method=='freeze_boundary' else [1.,.25,.0625,.015625,.00390625,0.]
                    for alpha in scales:
                        trial=proposed.clone()
                        if method in ['block_backtrack','freeze_boundary']:trial[:,0]=before[:,0]+alpha*(proposed[:,0]-before[:,0])
                        else:trial=before+alpha*(proposed-before)
                        yy=suffix(vae,z[:,2:]+trial,state);forwards+=1
                        cmse=(yy[:,:2]-edit[:,5:7]).square().mean()
                        if cmse<=bound*1.0001:
                            chosen=trial;break
                    assert chosen is not None
                    delta.copy_(chosen)
                    curve.append({'step':step,'alpha':alpha,'core_relative_rmse':float((cmse/coreden).sqrt()),'suffix_relative_mse':float((yy[:,2:]-base[:,7:]).square().mean()/denom)})
            with torch.no_grad():
                zz=z.clone();zz[:,2:]+=delta;final,_=stream(vae,zz)
                cmse=(final[:,5:7]-edit[:,5:7]).square().mean();smse=(final[:,7:]-base[:,7:]).square().mean()
                assert cmse<=bound*1.001
                save({'sample':p.stem,'method':method,'initial_suffix_mse':float(denom),'final_suffix_mse':float(smse),'suffix_reduction_pct':float(100*(1-smse/denom)),'relative_core_rmse':float((cmse/coreden).sqrt()),'prefix_core_max':float((final[:,:7]-edit[:,:7]).abs().max()),'before_edit_max':float((final[:,:5]-edit[:,:5]).abs().max()),'freeze_control_class_floor':float(floor),'seconds':time.time()-tic,'forwards':forwards+1,'backwards':32,'curve':curve})
                torch.save({'latent':zz.cpu(),'video':final.cpu()},OUT/f'{p.stem}_{method}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
