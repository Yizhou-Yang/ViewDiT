#!/usr/bin/env python3
"""Single edited-decode variant: reference before boundary, live norm after it."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
OUT=Path('/root/viewdit/results/video/cog_online_boundary_norm')
ROOT=Path('/root/viewdit/data/boundary_holdout')


class OnlineBoundaryNorm:
    def __init__(self,model,reference,target_call=1):
        self.reference=reference;self.target_call=target_call;self.count={};self.handles=[]
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.GroupNorm):self.handles.append(m.register_forward_hook(self.hook(name)))

    def hook(self,name):
        def apply(m,args,out):
            x=args[0];call=self.count.get(name,0);self.count[name]=call+1
            if call>self.target_call:return out
            b,c,t,h,w=x.shape
            mu,rs=self.reference[(name,call)]
            fixed=((x.reshape(b,m.num_groups,-1)-mu)*rs).reshape_as(x)
            if m.weight is not None:fixed=fixed*m.weight.view(1,c,1,1,1)
            if m.bias is not None:fixed=fixed+m.bias.view(1,c,1,1,1)
            if call<self.target_call:return fixed
            assert t%2==0
            return torch.cat([fixed[:,:,:t//2],out[:,:,t//2:]],dim=2)
        return apply

    def reset(self):self.count={}
    def close(self):
        for h in self.handles:h.remove()


@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'protocol':{'clips':['bmx-trees','boat','bus','dog'],'amplitudes':[.4,1.],'frames':17,'size':[128,224],'warning':'Development after dual method; not held-out. No extra edited stats capture is used for output. Raw capture for evaluation only.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        files=sorted((ROOT/'JPEGImages'/clip).glob('*.jpg'))[:17]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((224,128)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode();base,rs=capture(vae,z)
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5;field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        norm=OnlineBoundaryNorm(vae.decoder,rs)
        norm.reset();identity=vae.decode(z).sample
        norm.close()
        for amplitude in [.4,1.]:
            ze=z.clone();ze[:,:,-1]+=amplitude*field[:,:,-1]
            raw=vae.decode(ze).sample;denom=(raw[:,:,13:]-base[:,:,13:]).square().mean().sqrt()
            norm=OnlineBoundaryNorm(vae.decoder,rs)
            try:norm.reset();y=vae.decode(ze).sample
            finally:norm.close()
            save({'clip':clip,'amplitude':amplitude,'prefix_max_vs_noop':float((y[:,:,:13]-identity[:,:,:13]).abs().max()),'identity_max':float((identity-base).abs().max()),'edit_relative_rmse':float((y[:,:,13:]-raw[:,:,13:]).square().mean().sqrt()/denom),'edit_amplitude_ratio':float((y[:,:,13:]-base[:,:,13:]).square().mean().sqrt()/denom),'edited_stat_capture_required':False})
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
