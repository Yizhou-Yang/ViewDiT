#!/usr/bin/env python3
"""Full finite-difference Jacobian in a small controlled instance, not global proof."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_local_rank')

@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'resolution':[32,48],'samples':['bear_latent_geometric_impulse','blackswan_latent_geometric_impulse'],'latent_block':2,'frame_offsets':[0,1,2,3],'fd_h':.01,'method':'central finite difference for ALL 384 variables, no coordinate subsampling','warning':'low-resolution full Jacobian; numerical rank at declared thresholds, no global injectivity or full-resolution impossibility claim'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for name in report['protocol']['samples']:
        d=torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=True)
        vid=d['videos']['base'].cuda();x=F.interpolate(vid.permute(1,0,2,3),size=(32,48),mode='bilinear',align_corners=False).permute(1,0,2,3)
        z=vae.model.encode(x.unsqueeze(0),vae.scale)[0].float();y,states=stream(vae,z);state=states[2]
        point=z[:,2:3].clone();n=point.numel();h=.01;columns=[];tic=time.time()
        y0=suffix(vae,point,state)
        for j in range(n):
            perturb=torch.zeros_like(point);perturb.view(-1)[j]=h
            yp=suffix(vae,point+perturb,state);ym=suffix(vae,point-perturb,state)
            columns.append(((yp-ym)/(2*h)).cpu())
        jac=torch.stack(columns,-1).cuda();spectra={};weak=None
        for frames in [1,2,3,4]:
            je=jac[:,:frames].reshape(-1,n).double();gram=je.T@je
            vals,vecs=torch.linalg.eigh(gram);sv=vals.clamp(min=0).sqrt()
            spectra[str(frames)]={'sigma_min':float(sv[0]),'sigma_max':float(sv[-1]),'condition':float(sv[-1]/sv[0].clamp(min=1e-12)),'rank_relative_1e_3':int((sv>sv[-1]*.001).sum()),'rank_relative_1e_4':int((sv>sv[-1]*.0001).sum()),'dimension':n}
            if frames==2:weak=vecs[:,0].float().reshape_as(point)
        tests=[]
        gen=torch.Generator(device='cpu').manual_seed(998)
        directions={'weakest_core':weak,'random':torch.randn(point.shape,generator=gen).cuda()}
        for kind,v in directions.items():
            v=v/v.norm()
            predicted=(jac@v.flatten()).float()
            for amplitude in [.001,.01,.1]:
                measured=suffix(vae,point+amplitude*v,state)-y0
                tests.append({'direction':kind,'amplitude':amplitude,'core_change_norm':float(measured[:,:2].norm()),'tail_change_norm':float(measured[:,2:].norm()),'linearization_relative_error':float((measured-amplitude*predicted).norm()/(amplitude*predicted).norm().clamp(min=1e-12))})
        j2=torch.stack([((suffix(vae,point+h*.5*weak,state)-suffix(vae,point-h*.5*weak,state))/h),((suffix(vae,point+h*weak,state)-suffix(vae,point-h*weak,state))/(2*h))])
        save({'sample':name,'shape':list(z.shape),'spectra':spectra,'weak_fd_halfstep_relative_difference':float((j2[0]-j2[1]).norm()/j2[0].norm()),'nonlinear_tests':tests,'seconds':time.time()-tic})
        torch.save({'jacobian':jac.cpu(),'spectra':spectra},OUT/(name+'.pt'))
    report['complete']=True;save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
