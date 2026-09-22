#!/usr/bin/env python3
"""Architecture audit: causality of the full decoder, not just causal convolutions."""
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
ROOT=Path('/root/viewdit/results/video/wan_real_pilot')
OUT=Path('/root/viewdit/results/video/cog_locality_probe')
WEIGHT=Path('/root/viewdit/weights/CogVAE/diffusion_pytorch_model.safetensors')

@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()=='a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e'
    vae=AutoencoderKLCogVideoX.from_pretrained(str(WEIGHT.parent),local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    default=vae.num_latent_frames_batch_size
    report={'complete':False,'protocol':{'resolution':[64,96],'clips':['bear','blackswan','car-roundabout','camel'],'dtype':'FP32','model':'official CogVideoX-2b VAE using installed diffusers0.31.0','default_latent_batch':default,'warning':'Changing temporal batch size changes normalization and is a diagnostic, not a free quality fix. Tiny-resolution mechanism test, no perceptual benchmark.'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        d=torch.load(ROOT/(clip+'_latent_geometric_impulse.pt'),map_location='cpu',weights_only=True)
        v=d['videos']['base'].cuda();x=F.interpolate(v.permute(1,0,2,3),size=(64,96),mode='bilinear',align_corners=False).permute(1,0,2,3).unsqueeze(0)
        z=vae.encode(x).latent_dist.mode();standard=vae.decode(z).sample
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5
        field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        for batch in sorted(set([default,4,5])):
            vae.num_latent_frames_batch_size=batch;y=vae.decode(z).sample
            assert y.shape==standard.shape
            responses=[];tic=time.time()
            for j in range(z.shape[2]):
                zz=z.clone();zz[:,:,j]+=.1*field[:,:,j]
                yy=vae.decode(zz).sample
                responses.append((yy-y).square().mean((0,1,3,4)).sqrt().cpu().tolist())
            save({'clip':clip,'latent_batch':batch,'shape':list(z.shape),'decoded_shape':list(y.shape),'same_latent_vs_default_mse':float((y-standard).square().mean()),'same_latent_vs_default_max':float((y-standard).abs().max()),'impulse_frame_rms':responses,'seconds':time.time()-tic,'reconstruction_mse':float((y-x).square().mean())})
        vae.num_latent_frames_batch_size=default
    report['complete']=True;save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
