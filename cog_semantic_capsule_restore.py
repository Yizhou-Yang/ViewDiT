#!/usr/bin/env python3
"""Real generated latent, serialize reference capsule, restore only prefix cone.

Shared edited latent retained. Old latent4 allowed as a strong baseline.
Full49-frame reference re-decode is unnecessary for both methods.
"""
import io
import json
import time
from pathlib import Path
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
from cog_context_capsule import CapsuleNorm
from cog_exact_memory import NoUnusedTemporalCache
SRC=Path('/root/viewdit/results/video/cog_semantic_bear49')
OUT=Path('/root/viewdit/results/video/cog_semantic_capsule_restore')


@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    patch=NoUnusedTemporalCache(vae)
    report={'complete':False,'protocol':{'source':'true CogVideoX49-frame generated latents, two rounds','protected_frames':13,'restore_input_latents':5,'dtype':'FP32','storage':'Actual torch serialized payload plus exact tensor payload reported separately','baseline':'Keep old latent4, not entire old49-frame suffix. Same prefix-only decode for both.','warning':'One content, same future rewritten twice; not independent statistical validation. Timing excludes shared original raw decode, encode, first capture and model loading.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for rnd in [1,2]:
        d=torch.load(SRC/f'bear_round{rnd}.pt',map_location='cpu',weights_only=True)
        z=d['source_latent'].cuda();ze=d['latent'].cuda()
        base=d['videos']['base'].cuda();raw=d['videos']['raw'].cuda()
        _,stats=capture(vae,z[:,:,:5])
        selected={k:tuple(t.cpu() for t in v) for k,v in stats.items() if k[1]==1}
        capsule_path=OUT/f'round{rnd}_capsule.pt';torch.save({'schema':{'model':'THUDM/CogVideoX-2b/vae','diffusers':'0.31.0','boundary':13,'height':480,'width':720,'latent_batch_size':2},'moments':selected},capsule_path)
        loaded=torch.load(capsule_path,map_location='cuda',weights_only=True)['moments']
        payload=sum(t.numel()*t.element_size() for v in loaded.values() for t in v)
        old=z[:,:,4].clone();del z,stats,selected
        for method in ['capsule_fp32','old_latent_fp32','old_latent_fp16']:
            torch.cuda.synchronize();tic=time.perf_counter()
            candidate=ze[:,:,:5].clone()
            control=None
            if method=='capsule_fp32':
                control=CapsuleNorm(vae.decoder,loaded,'fp32');nbytes=payload;disk=capsule_path.stat().st_size
            else:
                dtype=torch.float32 if method.endswith('fp32') else torch.float16
                stored=old.to(dtype);nbytes=stored.numel()*stored.element_size()
                serialized=io.BytesIO();torch.save(stored.cpu(),serialized);disk=serialized.tell()
                candidate[:,:,4]=stored.float()
            try:
                y=vae.decode(candidate).sample[:,:,:13]
            finally:
                if control:control.close()
            result=raw.clone();result[:,:,:13]=y
            torch.cuda.synchronize();seconds=time.perf_counter()-tic
            error=float((y-base[:,:,:13]).abs().max())
            save({'round':rnd,'method':method,'tensor_payload_bytes':nbytes,'serialized_bytes':disk,'prefix_max':error,'edited_max':float((result[:,:,13:]-raw[:,:,13:]).abs().max()),'restore_seconds':seconds,'rendered_frames':49,'decoded_restore_frames':17,'decoded_restore_latents':5})
            if method=='capsule_fp32':torch.save({'video':result.cpu()},OUT/f'round{rnd}_restored.pt')
    patch.close();report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
