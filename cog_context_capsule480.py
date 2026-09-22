#!/usr/bin/env python3
"""Rate-distortion audit of boundary-only reference moments vs residual storage.

Protect outputs9:13 when changing latent4; earlier decoder call is unchanged.
Store only reference moments of affected call1, not all reference statistics.
"""
import io
import json
import time
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
ROOT=Path('/root/viewdit/data/v3_validation_data')
OUT=Path('/root/viewdit/results/video/cog_context_capsule480')


class CapsuleNorm:
    def __init__(self,model,stats,precision):
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
        self.stats={k:tuple(t.to(dtype).float() for t in v) for k,v in stats.items() if k[1]==1}
        self.count={};self.handles=[]
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.GroupNorm):self.handles.append(m.register_forward_hook(self.hook(name)))

    def hook(self,name):
        def apply(m,args,out):
            call=self.count.get(name,0);self.count[name]=call+1
            if call!=1:return out
            x=args[0];b,c,t,h,w=x.shape;mu,rs=self.stats[(name,call)]
            y=((x.reshape(b,m.num_groups,-1)-mu)*rs).reshape_as(x)
            if m.weight is not None:y=y*m.weight.view(1,c,1,1,1)
            if m.bias is not None:y=y+m.bias.view(1,c,1,1,1)
            assert t%2==0
            return torch.cat([y[:,:,:t//2],out[:,:,t//2:]],dim=2)
        return apply

    def close(self):
        for h in self.handles:h.remove()


def packed(t,dtype=torch.float32):
    t=t.detach().cpu().to(dtype).contiguous()
    return zlib.compress(t.view(torch.uint8).numpy().tobytes(),level=9)


@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'protocol':{'clips':['bear','blackswan'],'frames':17,'size':[480,720],'amplitudes':[.4,1.],'warning':'Development-only representation storage audit. Not a learned codec, semantic benchmark, or proof of optimal compression. Count payload bytes separately from shared schema/model. Pixel residual must be regenerated per candidate, reference context/capsule reusable.'},'rows':[]}
    def save(r=None):
        if r is not None:report['rows'].append(r);print('ROW',json.dumps(r),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        files=sorted(p for p in (ROOT/'JPEGImages'/clip).glob('*.jpg') if not p.name.startswith('.'))[:17]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((720,480)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode();base,rs=capture(vae,z)
        capsule=torch.cat([t.flatten() for k,v in rs.items() if k[1]==1 for t in v])
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5;field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        for amplitude in [.4,1.]:
            ze=z.clone();ze[:,:,-1]+=amplitude*field[:,:,-1]
            raw=vae.decode(ze).sample;denom=(raw[:,:,13:]-base[:,:,13:]).square().mean().sqrt()
            common={'clip':clip,'amplitude':amplitude,'reusable_capsule_scalars':capsule.numel(),'minimum_original_suffix_latent_scalars':z[:,:,-1].numel()}
            for precision in ['fp32','fp16','bf16']:
                control=CapsuleNorm(vae.decoder,rs,precision)
                try:y=vae.decode(ze).sample
                finally:control.close()
                dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
                save(common|{'method':'capsule_'+precision,'payload_bytes':capsule.numel()*torch.tensor([],dtype=dtype).element_size(),'zlib_payload_bytes':len(packed(capsule,dtype)),'prefix_max':float((y[:,:,:13]-base[:,:,:13]).abs().max()),'prefix_rmse':float((y[:,:,:13]-base[:,:,:13]).square().mean().sqrt()),'edit_relative_rmse':float((y[:,:,13:]-raw[:,:,13:]).square().mean().sqrt()/denom)})
            save(common|{'method':'minimum_original_suffix_latent_fp32','payload_bytes':z[:,:,-1].numel()*4,'zlib_payload_bytes':len(packed(z[:,:,-1])),'prefix_max':0.,'prefix_rmse':0.,'edit_relative_rmse':0.,'construction':'Retain original final latent; together with edited prefix exactly reconstruct original latent for reference decode. Pixel splice retains raw edited suffix.'})
            quantized=z.clone();quantized[:,:,-1]=z[:,:,-1].half().float()
            qref=vae.decode(quantized).sample
            save(common|{'method':'minimum_original_suffix_latent_fp16','payload_bytes':z[:,:,-1].numel()*2,'zlib_payload_bytes':len(packed(z[:,:,-1],torch.float16)),'prefix_max':float((qref[:,:,:13]-base[:,:,:13]).abs().max()),'prefix_rmse':float((qref[:,:,:13]-base[:,:,:13]).square().mean().sqrt()),'edit_relative_rmse':0.})
            residual=base[:,:,9:13]-raw[:,:,9:13]
            for dtype in [torch.float32,torch.float16]:
                restored=residual.to(dtype).float()+raw[:,:,9:13]
                save(common|{'method':'pixel_residual_'+str(dtype),'payload_bytes':residual.numel()*torch.tensor([],dtype=dtype).element_size(),'zlib_payload_bytes':len(packed(residual,dtype)),'prefix_max':float((restored-base[:,:,9:13]).abs().max()),'prefix_rmse':float((restored-base[:,:,9:13]).square().mean().sqrt()),'edit_relative_rmse':0.})
            v=residual[0].permute(1,0,2,3)
            for size in [(4,7),(8,14),(16,28),(32,56)]:
                coeff=F.interpolate(v,size=size,mode='area').half()
                correction=F.interpolate(coeff.float(),size=(480,720),mode='bilinear',align_corners=False).permute(1,0,2,3).unsqueeze(0)
                restored=raw[:,:,9:13]+correction
                save(common|{'method':'pixel_residual_grid_'+str(size),'payload_bytes':coeff.numel()*2,'zlib_payload_bytes':len(packed(coeff,torch.float16)),'prefix_max':float((restored-base[:,:,9:13]).abs().max()),'prefix_rmse':float((restored-base[:,:,9:13]).square().mean().sqrt()),'edit_relative_rmse':0.})
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
