#!/usr/bin/env python3
"""Remove unused k_t=1 cache clones; verify exact output vs installed0.31.0."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d


class NoUnusedTemporalCache:
    def __init__(self,vae):
        self.original=[]
        for m in vae.modules():
            if isinstance(m,CogVideoXCausalConv3d) and m.time_kernel_size==1:
                self.original.append((m,m.forward))
                m.forward=self.forward_for(m)

    @staticmethod
    def forward_for(m):
        def forward(inputs,conv_cache=None):
            padding=(m.width_pad,m.width_pad,m.height_pad,m.height_pad)
            return m.conv(F.pad(inputs,padding,mode='constant',value=0)),None
        return forward

    def close(self):
        for m,original in self.original:m.forward=original


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    out=Path('/root/viewdit/results/video/cog_exact_memory');out.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'rows':[],'protocol':'Installeddiffusers0.31.0 causalConv timekernel1 ignores conv_cache but clones full input. Remove only these unused clones, leave all arithmetic unchanged. This is engineering, not a research contribution.'}
    for h,w in [(128,224),(480,720)]:
        files=sorted(p for p in Path('/root/viewdit/data/v3_validation_data/JPEGImages/bear').glob('*.jpg') if not p.name.startswith('.'))[:17]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((w,h)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        z=vae.encode(x).latent_dist.mode();base=vae.decode(z).sample
        original_peak=torch.cuda.max_memory_allocated()
        patch=NoUnusedTemporalCache(vae)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        zz=vae.encode(x).latent_dist.mode();y=vae.decode(zz).sample
        patched_peak=torch.cuda.max_memory_allocated()
        latent_error=float((zz-z).abs().max());pixel_error=float((y-base).abs().max())
        count=len(patch.original);patch.close()
        assert latent_error==0 and pixel_error==0
        row={'height':h,'width':w,'frames':17,'patched_modules':count,'latent_max_error':latent_error,'pixel_max_error':pixel_error,'original_peak_allocated_bytes':original_peak,'patched_peak_allocated_bytes':patched_peak,'note':'patched peak includes retained base,z tensors; conservative compared to original.'}
        report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        (out/'stats.json').write_text(json.dumps(report,indent=2))
        del x,z,zz,y,base;torch.cuda.empty_cache()
    report['complete']=True;(out/'stats.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)


if __name__=='__main__':main()
