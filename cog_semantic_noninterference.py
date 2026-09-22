#!/usr/bin/env python3
"""Real CogVideoX-2b V2V denoising, shared-latent decoder audit.

Explicit FP32 VAE mode encoding, FP16 DiT, precomputed FP32 T5 embeddings.
No training, no posterior sampling confound, no silent resolution fallback.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXVideoToVideoPipeline, CogVideoXDDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from cog_boundary_moment_transport import capture, BoundaryMomentTransport
from cog_exact_memory import NoUnusedTemporalCache
from cog_context_capsule import CapsuleNorm
MODEL=Path('/root/viewdit/weights/CogVideoX-2b')
OUT=Path('/root/viewdit/results/video/cog_semantic_noninterference')
CASES={
    'bear': ['A white polar bear walking on rocky ground, realistic wildlife footage.', 'A black and white panda walking on rocky ground, realistic wildlife footage.'],
    'blackswan': ['A white swan swimming on a pond, realistic wildlife footage.', 'A white swan swimming on a pond under warm golden sunset light, realistic wildlife footage.'],
    'bus': ['A bright red bus driving on a city street, realistic video.', 'A bright yellow bus driving on a city street, realistic video.'],
    'boat': ['A red sailboat sailing on water, realistic video.', 'A red sailboat sailing on water under warm golden sunset light, realistic video.'],
}


def sync():
    torch.cuda.synchronize()


def load_video(clip, frames, height, width):
    root=Path('/root/viewdit/data/v3_validation_data')
    files=sorted(p for p in (root/'JPEGImages'/clip).glob('*.jpg') if not p.name.startswith('.'))
    assert len(files)>=frames, f'{clip}: only {len(files)} source frames, requested {frames}'
    arrays=[np.asarray(Image.open(p).convert('RGB').resize((width,height)),dtype=np.float32)/127.5-1 for p in files[:frames]]
    return torch.from_numpy(np.stack(arrays)).permute(3,0,1,2).unsqueeze(0).cuda()


def render(videos,path):
    keys=list(videos)
    frames=[]
    for t in range(next(iter(videos.values())).shape[2]):
        panel=Image.new('RGB',(320*len(keys),236),'white');draw=ImageDraw.Draw(panel)
        for j,k in enumerate(keys):
            a=videos[k][0,:,t].float().clamp(-1,1).permute(1,2,0)
            im=Image.fromarray(((a+1)*127.5).round().byte().numpy()).resize((320,212))
            panel.paste(im,(320*j,24));draw.text((320*j+4,4),k+' f'+str(t),fill='black')
        frames.append(panel)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=125,loop=0)
    select=[0,len(frames)//2,len(frames)-2,len(frames)-1]
    sheet=Image.new('RGB',(frames[0].width,236*len(select)),'white')
    for i,t in enumerate(select):sheet.paste(frames[t],(0,i*236))
    sheet.save(path.with_suffix('.png'))


@torch.no_grad()
def embeddings(clips, cache):
    if cache.exists():
        result=torch.load(cache,map_location='cpu',weights_only=True)
        assert all(p in result for c in clips for p in CASES[c]) and '' in result
        return result
    tokenizer=T5Tokenizer.from_pretrained(MODEL/'tokenizer',local_files_only=True)
    encoder=T5EncoderModel.from_pretrained(MODEL/'text_encoder',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    result={}
    for text in ['']+[p for c in clips for p in CASES[c]]:
        tokens=tokenizer(text,padding='max_length',max_length=226,truncation=True,add_special_tokens=True,return_tensors='pt')
        encoded=encoder(tokens.input_ids.cuda())[0]
        assert torch.isfinite(encoded).all()
        result[text]=encoded.half().cpu()
        print('EMBED',text,flush=True)
    torch.save(result,cache)
    del encoder;gc.collect();torch.cuda.empty_cache()
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--clips',default='bear')
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--width',type=int,default=720)
    parser.add_argument('--frames',type=int,default=17)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--strength',type=float,default=.6)
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--skip-roundtrip',action='store_true')
    parser.add_argument('--start-latent',type=int,default=4)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    clips=args.clips.split(',')
    embed=embeddings(clips,args.out/'embeddings.pt')
    vae=AutoencoderKLCogVideoX.from_pretrained(MODEL/'vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval().requires_grad_(False)
    memory_patch=NoUnusedTemporalCache(vae)
    dit=CogVideoXTransformer3DModel.from_pretrained(MODEL/'transformer',torch_dtype=torch.float16,local_files_only=True).eval().requires_grad_(False)
    scheduler=CogVideoXDDIMScheduler.from_pretrained(MODEL/'scheduler',local_files_only=True)
    pipe=CogVideoXVideoToVideoPipeline(tokenizer=None,text_encoder=None,vae=vae,transformer=dit,scheduler=scheduler)
    report={'complete':False,'protocol':vars(args)|{'out':str(args.out),'seed':20260921,'precision':'FP32 VAE/T5; FP16 DiT','sampler':'CogVideoX DDIM eta0','offload':'DiT CPU during FP32 VAE phases; transfer costs excluded from stage timers, not total wall time','sampling':'mode encode; same explicit noise; prefix pins after every step','warning':'17-frame run is an engineering pilot, not official49-frame quality. Shared true DiT latent; not trained instruction editor. Two rounds if requested reuse same latent for compositor and our decoder, not forced reencoding.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        p=args.out/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(args.out/'stats.json')
    save()
    for clip in clips:
        x=load_video(clip,args.frames,args.height,args.width)
        sync();tic=time.perf_counter();z=vae.encode(x).latent_dist.mode();sync();encode_sec=time.perf_counter()-tic
        del x
        base,ref_stats=capture(vae,z)
        z_current=z.clone()
        latent_frames=z.shape[2]
        assert latent_frames>=5 and latent_frames%2==1
        assert args.start_latent>=4 and args.start_latent%2==0 and args.start_latent<latent_frames
        target_call=(args.start_latent-2)//2
        protected_end=1+4*(args.start_latent-1)
        for rnd in range(args.rounds):
            prompt=CASES[clip][rnd]
            source=z_current.permute(0,2,1,3,4).to(torch.float16)*vae.config.scaling_factor
            scheduler.set_timesteps(args.steps,device='cuda')
            used_steps=scheduler.timesteps[args.steps-int(args.steps*args.strength):]
            assert len(used_steps)>0
            generator=torch.Generator(device='cuda').manual_seed(20260921+rnd)
            noise=torch.randn(source.shape,device='cuda',dtype=source.dtype,generator=generator)
            initial=scheduler.add_noise(source,noise,used_steps[:1])
            def pin(p,i,t,kw):
                value=kw['latents']
                pinned=source if i==len(used_steps)-1 else scheduler.add_noise(source,noise,used_steps[i+1:i+2])
                value[:,:args.start_latent]=pinned[:,:args.start_latent]
                return {'latents':value}
            dit.cuda()
            torch.cuda.reset_peak_memory_stats();sync();tic=time.perf_counter()
            latent=pipe(latents=initial.clone(),prompt_embeds=embed[prompt].cuda(),negative_prompt_embeds=embed[''].cuda(),height=args.height,width=args.width,num_inference_steps=args.steps,strength=args.strength,guidance_scale=6.,eta=0.,generator=generator,output_type='latent',callback_on_step_end=pin).frames
            sync();dit_sec=time.perf_counter()-tic;peak=torch.cuda.max_memory_allocated()
            assert torch.isfinite(latent).all(),'nonfinite DiT'
            dit.cpu();gc.collect();torch.cuda.empty_cache()
            ze=latent.float().permute(0,2,1,3,4)/vae.config.scaling_factor
            ze[:,:,:args.start_latent]=z[:,:,:args.start_latent]
            raw,edited_stats=capture(vae,ze)
            assert torch.isfinite(raw).all()
            videos={'base':base.cpu(),'raw':raw.cpu()}
            pixel=raw.clone();pixel[:,:,:protected_end]=base[:,:,:protected_end];videos['pixel_splice']=pixel.cpu()
            candidates={'raw':raw,'pixel_splice':pixel}
            extra={}
            for method in ['reference','dual']:
                control=BoundaryMomentTransport(vae.decoder,ref_stats,edited_stats,target_call=target_call,mode=method)
                try:
                    sync();tic=time.perf_counter();control.reset();y=vae.decode(ze).sample;sync()
                    extra[method]=time.perf_counter()-tic
                finally:control.close()
                candidates[method]=y;videos[method]=y.cpu()
            assert target_call==1, 'capsule compositor currently validates boundary13 only'
            capsule=CapsuleNorm(vae.decoder,ref_stats,'fp32')
            try:
                sync();tic=time.perf_counter();restored=vae.decode(ze).sample;sync()
                extra['capsule_compositor']=time.perf_counter()-tic
            finally: capsule.close()
            compact_pixel=raw.clone();compact_pixel[:,:,:protected_end]=restored[:,:,:protected_end]
            candidates['capsule_compositor']=compact_pixel;videos['capsule_compositor']=compact_pixel.cpu()
            del restored
            denom=(raw[:,:,protected_end:]-base[:,:,protected_end:]).square().mean().sqrt().clamp(min=1e-12)
            for method,y in candidates.items():
                recost=None;rt_mse=None;latent_drift=None
                if not args.skip_roundtrip:
                    sync();tic=time.perf_counter();encoded=vae.encode(y.clamp(-1,1)).latent_dist.mode();sync();recost=time.perf_counter()-tic
                    rt=vae.decode(encoded).sample
                    rt_mse=float((rt-y).square().mean())
                    latent_drift=float((encoded-ze).square().mean().sqrt()/ze.square().mean().sqrt())
                row={'clip':clip,'round':rnd+1,'method':method,'prompt':prompt,'effective_dit_steps':len(used_steps),'dit_cfg_forward_calls':len(used_steps),'dit_seconds':dit_sec,'dit_peak_allocated_bytes':peak,'input_encode_seconds':encode_sec,'extra_decode_seconds':extra.get(method,0.),'protected_max':float((y[:,:,:protected_end]-base[:,:,:protected_end]).abs().max()),'edit_relative_rmse_vs_raw':float((y[:,:,protected_end:]-raw[:,:,protected_end:]).square().mean().sqrt()/denom),'edit_rms':float((y[:,:,protected_end:]-base[:,:,protected_end:]).square().mean().sqrt()),'roundtrip_mse':rt_mse,'reencode_latent_relative_rms':latent_drift,'reencode_seconds':recost,'shared_latent_with_pixel_reuse':True,'capsule_payload_bytes':9472 if method=='capsule_compositor' else None}
                save(row)
            torch.save({'videos':videos,'latent':ze.cpu(),'source_latent':z.cpu(),'prompt':prompt},args.out/f'{clip}_round{rnd+1}.pt')
            render(videos,args.out/f'{clip}_round{rnd+1}.gif')
            z_current=ze
            print('COMPOSITOR_REUSE_EQUIVALENCE: next-round input latent identical by construction; decoder-side method cannot claim unique latent reuse advantage.',flush=True)
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
