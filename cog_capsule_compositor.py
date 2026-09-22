#!/usr/bin/env python3
"""Compact reference-context compositing without retaining old future latent.

Accept pixel compositing as the exact output operation, and benchmark its
reference representation. This avoids claiming superiority to splice by error.
"""
import json
import time
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
from cog_context_capsule import CapsuleNorm, packed
ROOT=Path('/root/viewdit/data/v3_validation_data')
OUT=Path('/root/viewdit/results/video/cog_capsule_compositor')


def lowrank(block,rank):
    matrix=block.flatten(0,1).float()
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    r=min(rank,len(s))
    left=(u[:,:r]*s[:r]).half()
    right=vh[:r].half()
    restored=(left.float()@right.float()).reshape_as(block)
    return restored,left.numel()*2+right.numel()*2,len(packed(left,torch.float16))+len(packed(right,torch.float16))


@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'protocol':{'clips':['bear','blackswan'],'size':[480,720],'frames':17,'amplitude':.4,'warning':'Exploratory480P only2 contents. A reference representation for pixel composition, not a new semantic editor. Raw baseline/source setup excluded from restore timing. Float tolerance1e-5, not bit-exact. Reference old latent recovery baselines quantized and low-rank retained.'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        files=sorted(p for p in (ROOT/'JPEGImages'/clip).glob('*.jpg') if not p.name.startswith('.'))[:17]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((720,480)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode();del x
        base,rs=capture(vae,z)
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5;field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        ze=z.clone();ze[:,:,-1]+=.4*field[:,:,-1]
        raw=vae.decode(ze).sample
        oracle=raw.clone();oracle[:,:,:13]=base[:,:,:13]
        capsule=torch.cat([t.flatten() for k,v in rs.items() if k[1]==1 for t in v])
        spec=[('capsule_fp32',None,9472,len(packed(capsule))),('original_latent_fp32',z[:,:,-1],z[:,:,-1].numel()*4,len(packed(z[:,:,-1]))),('original_latent_fp16',z[:,:,-1].half().float(),z[:,:,-1].numel()*2,len(packed(z[:,:,-1],torch.float16)))]
        old=z[:,:,-1]
        scale=old.abs().amax(dim=(-2,-1),keepdim=True)/127
        quant=(old/scale.clamp(min=1e-8)).round().clamp(-127,127).to(torch.int8)
        restored=quant.float()*scale
        spec.append(('original_latent_int8',restored,quant.numel()+scale.numel()*4,len(zlib.compress(quant.cpu().numpy().tobytes()))+len(packed(scale))))
        for rank in [1,2,4,8,16]:
            restored,nbytes,zbytes=lowrank(old[0],rank)
            spec.append((f'original_latent_rank{rank}',restored.unsqueeze(0),nbytes,zbytes))
        for method,restore,nbytes,zbytes in spec:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter()
            if restore is None:
                norm=CapsuleNorm(vae.decoder,rs,'fp32')
                try:reference=vae.decode(ze).sample
                finally:norm.close()
            else:
                zz=ze.clone();zz[:,:,-1]=restore
                reference=vae.decode(zz).sample
            result=raw.clone();result[:,:,:13]=reference[:,:,:13]
            torch.cuda.synchronize();seconds=time.perf_counter()-tic
            error=float((result-oracle).abs().max())
            save({'clip':clip,'method':method,'payload_bytes':nbytes,'zlib_payload_bytes':zbytes,'max_error_vs_oracle':error,'prefix_rmse':float((result[:,:,:13]-oracle[:,:,:13]).square().mean().sqrt()),'edited_max_error':float((result[:,:,13:]-raw[:,:,13:]).abs().max()),'passes_1e5':error<=1e-5,'reference_restore_seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
