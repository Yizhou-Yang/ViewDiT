#!/usr/bin/env python3
"""Reference-moment replay removes a global normalization path without retraining."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/cog_norm_replay')

class ReplayNorm:
    def __init__(self,model):
        self.mode='capture';self.stats={};self.count={};self.handles=[]
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.GroupNorm):self.handles.append(m.register_forward_hook(self.make_hook(name)))
    def make_hook(self,name):
        def hook(m,args,out):
            x=args[0];i=self.count.get(name,0);self.count[name]=i+1
            if x.ndim!=5:return out
            b,c,t,h,w=x.shape;g=m.num_groups
            if self.mode=='capture':
                v=x.reshape(b,g,-1);mu=v.mean(-1,keepdim=True);rs=(v.var(-1,unbiased=False,keepdim=True)+m.eps).rsqrt()
                self.stats[(name,i)]=(mu,rs);return out
            if self.mode=='framewise':
                v=x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
                return F.group_norm(v,g,m.weight,m.bias,m.eps).reshape(b,t,c,h,w).permute(0,2,1,3,4)
            mu,rs=self.stats[(name,i)]
            y=((x.reshape(b,g,-1)-mu)*rs).reshape_as(x)
            if m.weight is not None:y=y*m.weight.view(1,c,1,1,1)
            if m.bias is not None:y=y+m.bias.view(1,c,1,1,1)
            return y
        return hook
    def reset(self,mode):self.mode=mode;self.count={}
    def close(self):
        for h in self.handles:h.remove()

@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'protocol':{'clips':['bear','blackswan','car-roundabout','camel'],'resolution':[64,96],'latent_batch':'default','edits':'last latent geometric perturbation, amplitudes0.1 and0.4','warning':'moments side information changes decoder execution; not standard native latent equivalence; untested perceptual and downstream DiT quality'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        d=torch.load(ROOT/(clip+'_latent_geometric_impulse.pt'),map_location='cpu',weights_only=True);v=d['videos']['base'].cuda()
        x=F.interpolate(v.permute(1,0,2,3),size=(64,96),mode='bilinear',align_corners=False).permute(1,0,2,3).unsqueeze(0)
        z=vae.encode(x).latent_dist.mode();original=vae.decode(z).sample
        norm=ReplayNorm(vae.decoder);norm.reset('capture');ref=vae.decode(z).sample
        norm.reset('replay');replayed=vae.decode(z).sample
        assert (ref-original).abs().max()==0
        norm.reset('framewise');framewise_ref=vae.decode(z).sample
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5;field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        for amplitude in [.1,.4]:
            ze=z.clone();ze[:,:,-1]+=amplitude*field[:,:,-1]
            norm.reset('replay');fixed=vae.decode(ze).sample
            norm.reset('framewise');framewise_edit=vae.decode(ze).sample
            norm.close();raw=vae.decode(ze).sample
            torch.save({'videos':{'base':original.cpu(),'raw':raw.cpu(),'moment_replay':fixed.cpu(),'framewise':framewise_edit.cpu()},'moments':{k:tuple(t.cpu() for t in v) for k,v in norm.stats.items()}},OUT/f'{clip}_{amplitude}.pt')
            norm=ReplayNorm(vae.decoder);norm.reset('capture');vae.decode(z)
            denom=(raw[:,:,13:]-original[:,:,13:]).square().mean().sqrt()
            save({'clip':clip,'amplitude':amplitude,'identity_max':float((replayed-original).abs().max()),'identity_mse':float((replayed-original).square().mean()),'raw_earlier_rms':float((raw[:,:,:13]-original[:,:,:13]).square().mean().sqrt()),'raw_earlier_max':float((raw[:,:,:13]-original[:,:,:13]).abs().max()),'replay_earlier_rms':float((fixed[:,:,:13]-replayed[:,:,:13]).square().mean().sqrt()),'replay_earlier_max':float((fixed[:,:,:13]-replayed[:,:,:13]).abs().max()),'edited_chunk_vs_raw_relative_rmse':float((fixed[:,:,13:]-raw[:,:,13:]).square().mean().sqrt()/denom),'edit_amplitude_ratio':float((fixed[:,:,13:]-replayed[:,:,13:]).square().mean().sqrt()/denom),'framewise_identity_mse':float((framewise_ref-original).square().mean()),'framewise_earlier_rms':float((framewise_edit[:,:,:13]-framewise_ref[:,:,:13]).square().mean().sqrt()),'moment_scalars':sum(a.numel()+b.numel() for a,b in norm.stats.values()),'latent_scalars':z.numel()})
        norm.close()
    report['complete']=True;save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
