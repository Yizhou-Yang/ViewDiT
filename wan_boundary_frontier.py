#!/usr/bin/env python3
"""Within-token boundary control. Sidecar is NOT a standard reusable DiT latent."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from wan_vae_official import WanVAE, Resample
from wan_state_probe import stream
from wan_tail_compensate import suffix

ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/wan_boundary_frontier')

class PhaseSidecar:
    def __init__(self, vae, z, boundary, rank=8, basis='random'):
        self.vae=vae;self.boundary=boundary;self.rank=rank;self.chunk=0;self.coeff=None;self.records=[]
        layers=[m for m in vae.model.decoder.modules() if isinstance(m,Resample) and m.mode=='upsample3d']
        self.layer=layers[-1]
        self.handle=self.layer.register_forward_hook(self.hook)
        with torch.no_grad():stream(vae,z)
        channels=self.records[-1].shape[1]
        gen=torch.Generator(device='cpu').manual_seed(741)
        if basis=='random':
            q=torch.linalg.qr(torch.randn(channels,rank,generator=gen)).Q.cuda()
        else:
            f=torch.cat(self.records,2)[0].permute(1,2,3,0).reshape(-1,channels).float()
            gram=f.T@f/f.shape[0]
            q=torch.linalg.eigh(gram.double()).eigenvectors[:,-rank:].float()
        self.q=q.detach();self.feature_scale=torch.cat(self.records,2).square().mean().sqrt().detach()
        self.records=[];self.chunk=0
        self.grid=z.shape[-2:];self.nframes=17-boundary
    def hook(self,module,args,out):
        start=0 if self.chunk==0 else 1+4*(self.chunk-1)
        self.chunk+=1
        if self.coeff is None:
            self.records.append(out.detach())
            return out
        first=max(start,self.boundary);last=min(start+out.shape[2],17)
        if first>=last:return out
        c=self.coeff[:,first-self.boundary:last-self.boundary]
        c=F.interpolate(c.permute(1,0,2,3),size=out.shape[-2:],mode='bilinear',align_corners=False).permute(1,0,2,3)
        delta=torch.einsum('cr,rthw->cthw',self.q,c)*self.feature_scale
        return out+F.pad(delta.unsqueeze(0),(0,0,0,0,first-start,start+out.shape[2]-last))
    def decode(self,z,coeff,state):
        self.chunk=2;self.coeff=coeff
        return suffix(self.vae,z[:,2:],state)
    def close(self):
        self.handle.remove()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=8);parser.add_argument('--boundary',type=int,default=7);parser.add_argument('--steps',type=int,default=32);parser.add_argument('--out',default=str(OUT));args=parser.parse_args()
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    vae=WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth',dtype=torch.float32,device='cuda')
    report={'complete':False,'protocol':{'boundary':args.boundary,'steps':args.steps,'lr':.03,'basis_seed':741,'rank':8,'feature_location':'after last temporal upsampling, native 4-frame expanded feature','loss':'suffix MSE / initial, + lambda core MSE/core edit energy for native optimization','warning':'sidecar changes representation and needs injection at decoding; NOT native DiT latent; pixel residual is a fair and necessary baseline; oracle pixel splice exact by construction'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        tmp=outdir/'stats.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(outdir/'stats.json')
    for p in sorted(ROOT.glob('*.pt'))[:args.limit]:
        d=torch.load(p,map_location='cpu',weights_only=True);z=d['edited_z'].cuda();base=d['videos']['base'].cuda();edit=d['videos']['edit_latent_only'].cuda();b=args.boundary;k=b-5
        with torch.no_grad():
            replay,states=stream(vae,z);assert (replay-edit).abs().max()<1e-6
            state=states[2];denom=(edit[:,b:]-base[:,b:]).square().mean().clamp(min=1e-12)
            coreden=(edit[:,5:b]-base[:,5:b]).square().mean().clamp(min=1e-12)
            target=edit.clone();target[:,b:]=base[:,b:]
            zr=vae.model.encode(target.unsqueeze(0),vae.scale)[0].float()
            zh=z.clone();zh[:,2:]=zr[:,2:]
            rec,_=stream(vae,zh)
        videos={'base':base.cpu(),'edit':edit.cpu(),'oracle_pixel_splice':target.cpu()}
        def record(name,y,seconds=0,params=0,extra=None):
            smse=float((y[:,b:]-base[:,b:]).square().mean());cmse=float((y[:,5:b]-edit[:,5:b]).square().mean())
            r={'sample':p.stem,'method':name,'suffix_mse':smse,'suffix_reduction_pct':100*(1-smse/float(denom)),'core_mse':cmse,'relative_core_rmse':math.sqrt(cmse/float(coreden)),'prefix_core_max':float((y[:,:b]-edit[:,:b]).abs().max()),'before_edit_max':float((y[:,:5]-edit[:,:5]).abs().max()),'core_edit_amplitude_ratio':float((y[:,5:b]-base[:,5:b]).norm()/(edit[:,5:b]-base[:,5:b]).norm()),'seconds':seconds,'control_scalars':params,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
            if extra:r.update(extra)
            videos[name]=y.cpu();save(r)
        record('no_compensation',edit);record('hybrid_reencode',rec);record('oracle_pixel_splice',target)
        for lam in [1.,10.,100.]:
            tic=time.time();torch.cuda.reset_peak_memory_stats();delta=torch.zeros_like(z[:,2:],requires_grad=True);opt=torch.optim.Adam([delta],lr=.03)
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True);y=suffix(vae,z[:,2:]+delta,state)
                loss=(y[:,k:]-base[:,b:]).square().mean()/denom+lam*(y[:,:k]-edit[:,5:b]).square().mean()/coreden+.001*delta.square().mean()
                assert torch.isfinite(loss);loss.backward();opt.step()
            with torch.no_grad():
                zz=z.clone();zz[:,2:]+=delta;yy,_=stream(vae,zz)
                record(f'native_lambda{int(lam)}',yy,time.time()-tic,delta.numel())
        for basis in ['random','pca']:
            ctrl=PhaseSidecar(vae,z,b,8,basis)
            coeff=torch.zeros(8,17-b,*ctrl.grid,device='cuda',requires_grad=True)
            tic=time.time();torch.cuda.reset_peak_memory_stats();opt=torch.optim.Adam([coeff],lr=.03)
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True);y=ctrl.decode(z,coeff,state)
                loss=(y[:,k:]-base[:,b:]).square().mean()/denom+.001*coeff.square().mean()
                assert torch.isfinite(loss);loss.backward();opt.step()
            with torch.no_grad():
                yy=torch.cat([edit[:,:5],ctrl.decode(z,coeff,state)],1)
                ctrl.chunk=0;ctrl.coeff=coeff;full,_=stream(vae,z)
                assert (full-yy).abs().max()<1e-5
                record('phase_sidecar_'+basis,full,time.time()-tic,coeff.numel(),{'standard_latent':False,'full_replay_max':float((full-yy).abs().max())})
                torch.save({'coeff':coeff.detach().cpu(),'q':ctrl.q.cpu(),'scale':ctrl.feature_scale.cpu(),'boundary':b},outdir/f'{p.stem}_{basis}_control.pt')
            ctrl.close()
        gh=int(math.sqrt((8*z.shape[-2]*z.shape[-1]/3)*(base.shape[-2]/base.shape[-1])))
        gw=int((8*z.shape[-2]*z.shape[-1]/3)/gh)
        delta=torch.zeros(17-b,3,gh,gw,device='cuda',requires_grad=True);opt=torch.optim.Adam([delta],lr=.03);tic=time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            r=F.interpolate(delta,size=base.shape[-2:],mode='bilinear',align_corners=False).permute(1,0,2,3)
            y=(edit[:,b:]+r).clamp(-1,1);loss=(y-base[:,b:]).square().mean()/denom+.001*delta.square().mean();loss.backward();opt.step()
        with torch.no_grad():
            yy=edit.clone();yy[:,b:]=(edit[:,b:]+F.interpolate(delta,size=base.shape[-2:],mode='bilinear',align_corners=False).permute(1,0,2,3)).clamp(-1,1)
            record('equal_budget_pixel_residual',yy,time.time()-tic,delta.numel(),{'grid':[gh,gw],'standard_latent':False})
        torch.save({'videos':videos,'boundary':b},outdir/f'{p.stem}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)

if __name__=='__main__':main()
